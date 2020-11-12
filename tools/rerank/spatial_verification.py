import copy
import io

import pydegensac
from scipy import spatial
import matplotlib.pyplot as plt
from skimage import feature
from skimage import io as skio


def compute_putative_matching_keypoints(test_keypoints,
                                        test_descriptors,
                                        train_keypoints,
                                        train_descriptors,
                                        use_ratio_test=True,
                                        matching_threshold=0.95,
                                        max_distance=0.95):
    """Finds matches from `test_descriptors` to KD-tree of `train_descriptors`."""
    train_descriptor_tree = spatial.cKDTree(train_descriptors)

    if use_ratio_test:
        distances,matches=train_descriptor_tree.query(
            test_descriptors,k=2,n_jobs=-1
        )
        test_kp_count = test_keypoints.shape[0]
        train_kp_count = train_keypoints.shape[0]
        test_matching_keypoints=np.array([
            test_keypoints[i,]
            for i in range(test_kp_count)
            if distances[i][0] < matching_threshold*distances[i][1]
        ])
        train_matching_keypoints=np.array([
            train_keypoints[matches[i][0],]
            for i in range(train_kp_count)
            if distances[i][0] < matching_threshold*distances[i][1]
        ])

    else:
        _, matches = train_descriptor_tree.query(
              test_descriptors, distance_upper_bound=max_distance)

        test_kp_count = test_keypoints.shape[0]
        train_kp_count = train_keypoints.shape[0]

        test_matching_keypoints = np.array([
              test_keypoints[i,]
              for i in range(test_kp_count)
              if matches[i] != train_kp_count
          ])
        train_matching_keypoints = np.array([
              train_keypoints[matches[i],]
              for i in range(test_kp_count)
              if matches[i] != train_kp_count
          ])
    return test_matching_keypoints, train_matching_keypoints 


def compute_num_inliers(test_keypoints, 
                        test_descriptors, 
                        train_keypoints,
                        train_descriptors,
                        use_ratio_test=False,
                        draw_matches=False,
                        query_im_array=None,
                        index_im_array=None):
    """Returns the number of RANSAC inliers."""
    test_match_kp, train_match_kp = \
            compute_putative_matching_keypoints(test_keypoints, 
                                                test_descriptors, 
                                                train_keypoints, 
                                                train_descriptors,
                                                use_ratio_test=use_ratio_test)
    if test_match_kp.shape[
          0] <= 4:  # Min keypoints supported by `pydegensac.findHomography()`
        return 0, b''

    try:
        _, mask = pydegensac.findHomography(test_match_kp, train_match_kp,
                                            MAX_REPROJECTION_ERROR,
                                            HOMOGRAPHY_CONFIDENCE,
                                            MAX_RANSAC_ITERATIONS)
    except np.linalg.LinAlgError:  # When det(H)=0, can't invert matrix.
        return 0, b''

    inliers = mask if mask is not None else []
    match_viz_bytes = b''
    if isinstance(query_im_array, np.ndarray) and isinstance(index_im_array, np.ndarray) and draw_matches:
        query_im_scale_factors = [1.0, 1.0]
        index_im_scale_factors = [1.0, 1.0]
        inlier_idxs = np.nonzero(inliers)[0]
        _, ax = plt.subplots()
        ax.axis('off')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        feature.plot_matches(
            ax,
            query_im_array,
            index_im_array,
            test_match_kp * query_im_scale_factors,
            train_match_kp * index_im_scale_factors,
            np.column_stack((inlier_idxs, inlier_idxs)),
            only_matches=False)

        match_viz_io = io.BytesIO()
        plt.savefig(match_viz_io, format='jpeg', bbox_inches='tight', pad_inches=0)
        match_viz_bytes = match_viz_io.getvalue()
    
    return int(copy.deepcopy(mask).astype(np.float32).sum()), match_viz_bytes


