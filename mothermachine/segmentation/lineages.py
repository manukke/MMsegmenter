import numpy as np
import sklearn.cluster as cluster
import resource

def predict_clusters(cents_clean, scan_eps = 6, min_samples = 4):
    """

    :param cents_clean:
    :param scan_eps:
    :param min_samples:
    :return:
    """

    ms = cluster.DBSCAN(eps=scan_eps, min_samples = min_samples, n_jobs=-1)
    predicted_clusters = ms.fit_predict(cents_clean)
    return predicted_clusters


def determine_position_of_cell_in_trench(props_sort):
    """
    determine whether the cells in a given trench are the mother, cell_pos=0,
    the cell right below the mother, cell_pos=1, cell below that, cell_pos=2, etc
    :param props_sort: properties sorted into lineages by sort_cells_into_lineages
    """
    # groupby lane, pos, lineage, time-point then sort cells 
    # top cell will be the mother_cell
    grouped = props_sort.groupby(['lane_num','pos_num','lineage_idx','t_frame'])
    num_cells = np.array(grouped.size())
    ac = np.array([np.arange(nc) for nc in num_cells])
    cell_pos = np.array([item for sublist in ac for item in sublist])
    props_sort['cell_pos'] = cell_pos
    return props_sort

def create_linear_index(props_sort):
    """
    create a unique index for each lineage in all lanes and positions in the dataframe
    :param props_sort: properties sorted into lineages by sort_cells_into_lineages
    """
    # have a linear idx for all cells in a group
    #linear_idx = props_sort.groupby(['pos_num','lineage_idx']).ngroup()
    linear_idx = props_sort.groupby(['lane_num','pos_num','lineage_idx']).ngroup()
    props_sort.loc[linear_idx.index,'linear_lineage_idx'] = np.array(linear_idx)
    return props_sort

def sort_cells_into_lineages(props, scan_eps = 6, min_samples = 4):
    """
    sort cells into lineages based on dbscan clustering of a z-stack of all centroids
    in data_frame for a given lane_num and pos_num
    :param props: a props pandas dataframe
    :param scan_eps: scan_epsilon parameter used by dbscan
    :param min_samples: minimum samples used by dbscan
    :return:
    """
    
    def cluster_it(prop):
        cents = np.transpose(np.vstack((prop.centx,prop.centy)))    
        predicted_clusters = predict_clusters(cents, scan_eps = scan_eps, min_samples=min_samples)
        prop['lineage_idx'] = predicted_clusters
        return prop

    props = props.groupby(['lane_num','pos_num']).apply(lambda x: cluster_it(x))

    # remove cells that aren't in a lineage
    props = props[props.lineage_idx > -1]
    props['centy_flipped'] = (props['centy']*props['trench_inversion_mult'])
    props_sort = props.sort_values(['lane_num','pos_num','lineage_idx','t_frame','centy_flipped'])
    
    return props_sort


def sort_cells_into_lineages_chunked(props, scan_eps = 12, min_samples =10, 
                             init_group_size = 300, trailing_group_size = 120):
    
    """
    experimental chunk sorting for very, very long time-traces were dbscan is too slow
    breaks time-series up into chunks, the first chunk is size 'init_group_size' 
    following chunks are size 'trailing_group_size' (currently this is not needed but in the future
    having a more stable first chunk could be useful)
    does dbscan on all the chunks then dbscan to group the chunks together
    could be improved, still a big hack
    """

    def get_group_idx(t_frames):
        uniq_t_frames = np.unique(t_frames)
        t_frames_trailing = uniq_t_frames[uniq_t_frames > init_group_size]
        t_splits = np.array_split(t_frames_trailing, (len(t_frames_trailing)//trailing_group_size)+1)
        t_splits.insert(0,uniq_t_frames[uniq_t_frames <= init_group_size])
        group_idx = [t_frames.isin(ts).values.astype(np.uint16)*i for i,ts in enumerate(t_splits)]
        group_idx = np.sum(group_idx,axis=0)
        return group_idx
    
    def cluster_it(prop):

        group_idx = get_group_idx(prop.t_frame)
        gb = prop.groupby(group_idx)
        print(np.unique(prop.pos_num))
        print("mem used: %2.3f" % (int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/(10**6)))
        init_clusters = []
        cluster_cents = []
        cluster_cent_idcs = []
        
        for group in gb:
            chunk = np.transpose(np.vstack((group[1].centx,group[1].centy)))
            predicted_clusters = predict_clusters(chunk, scan_eps = scan_eps, min_samples=min_samples)
            max_cluster_idx = np.max([0]+init_clusters)
            predicted_clusters[predicted_clusters > -1] = predicted_clusters[predicted_clusters > -1] + max_cluster_idx
            init_clusters.extend(predicted_clusters.tolist())
            for pred_cluster in np.unique(predicted_clusters):
                cluster_cents.append(np.mean(chunk[predicted_clusters == pred_cluster],axis=0))
                cluster_cent_idcs.append(pred_cluster)

        cluster_cent_idcs_arr = np.array(cluster_cent_idcs)
        cluster_cents_arr = np.array(cluster_cents)[cluster_cent_idcs_arr > -1]
        cluster_cent_idcs_arr = cluster_cent_idcs_arr[cluster_cent_idcs_arr > -1]

        clustered_clusters = predict_clusters(cluster_cents_arr[:,0].reshape(-1,1),min_samples=7,scan_eps=4)
        init_clusters = np.array(init_clusters)
        for i,clust in enumerate(cluster_cent_idcs_arr):
            init_clusters[init_clusters == clust] = clustered_clusters[i] 

        prop['lineage_idx'] = init_clusters
        return prop
        
        
    props = props.groupby(['lane_num','pos_num']).apply(lambda x: cluster_it(x))

    # remove cells that aren't in a lineage
    props = props[props.lineage_idx > -1]
    props['centy_flipped'] = (props['centy']*props['trench_inversion_mult'])
    props_sort = props.sort_values(['lane_num','pos_num','lineage_idx','t_frame','centy_flipped'])

    return props_sort
    
