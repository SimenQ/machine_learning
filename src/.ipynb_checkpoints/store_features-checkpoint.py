import pandas as pd
from scipy.spatial.distance import cdist

# def stores_density_per_location_by_type(stores_df, plaace_df, grunnkrets_df, geo="district_name", lv_desc="lv1_desc"):
#     """
#     Density of stores of the same type in a geographic location.

#     This depends on population
#     """
#     number_of_stores = store_types_count_by_geo_group(
#         stores_df, plaace_df, grunnkrets_df, geo=geo, lv_desc=lv_desc)['count']
#     population = 0
#     return number_of_stores / population

def stores_in_radius(stores_df, plaace_df, radius=0.1, store_type_group=None):
    """
    Number of stores within a given radius. Can also indicate category to filter.
    """
    mat = cdist(stores_df[['lat', 'lon']],
                stores_df[['lat', 'lon']], metric='euclidean')
    new_df = pd.DataFrame(
        mat, index=stores_df['store_id'], columns=stores_df['store_id'])

    if store_type_group is None:
        count = new_df[(new_df < radius) & (new_df > 0)].count(axis=1)
        return count.to_frame(name="count")

    else:
        combined_df = stores_df.merge(
            plaace_df, how="inner", on="plaace_hierarchy_id")
        test_df = new_df[(new_df < radius) & (new_df > 0)]
        store_count = {}

        for index, row in test_df.iterrows():
            nearby_stores = row.dropna().index.values
            index_type = combined_df[combined_df['store_id']
                                     == index][store_type_group].values[0]
            number_same = combined_df[(combined_df['store_id'].isin(nearby_stores)) & (
                combined_df[store_type_group] == index_type)]['store_id'].count()
            store_count[index] = number_same

        df = pd.DataFrame.from_dict(store_count, orient='index', columns=['count'])
        df.index.rename('store_id', inplace=True)
        return df

def store_types_count_by_geo_group(stores_df, plaace_df, grunnkrets_df, agg_name, geo_group="district_name", store_type_group="lv1_desc"):
    """
    Number of stores of the same type in a geographic location.
    """
    combined_df = stores_df.merge(plaace_df, how="inner", on="plaace_hierarchy_id").merge(
        grunnkrets_df, how="inner", on="grunnkrets_id")
    return combined_df.groupby(by=[geo_group, store_type_group])['store_id'].count().reset_index(name=agg_name)


def store_types_revenue_by_geo_group(stores_df, plaace_df, grunnkrets_df, agg_name, geo_group="district_name", store_type_group="lv1_desc"):
    """
    Total revenue of stores of the same type in a geographic location.
    """
    combined_df = stores_df.merge(plaace_df, how="inner", on="plaace_hierarchy_id").merge(
        grunnkrets_df, how="inner", on="grunnkrets_id")
    return combined_df.groupby(by=[geo_group, store_type_group])['revenue'].sum().reset_index(name=agg_name)

def store_types_all_count_by_geo_groups(stores_df, plaace_df, grunnkrets_df, store_types, geo_groups):
    merged_df = stores_df.merge(grunnkrets_df, how="left", on="grunnkrets_id").merge(plaace_df, how="left", on="plaace_hierarchy_id")
    
    df_list = []
    for geo_group in geo_groups:
        for store_type in store_types:
            df = store_types_count_by_geo_group(stores_df, plaace_df, grunnkrets_df, geo_group=geo_group, agg_name=f"{geo_group}_{store_type}", store_type_group=store_type)
            df_list.append(merged_df.merge(df, how="left", on=[geo_group, store_type])[['store_id', f"{geo_group}_{store_type}"]])
    
    dfs = [df.set_index('store_id') for df in df_list]
    return pd.concat(dfs, axis=1)

def store_types_all_revenue_by_geo_groups(stores_df, plaace_df, grunnkrets_df, store_types, geo_groups):
    merged_df = stores_df.merge(grunnkrets_df, how="left", on="grunnkrets_id").merge(plaace_df, how="left", on="plaace_hierarchy_id")
    
    df_list = []
    for geo_group in geo_groups:
        for store_type in store_types:
            df = store_types_revenue_by_geo_group(stores_df, plaace_df, grunnkrets_df, geo_group=geo_group, agg_name=f"{geo_group}_{store_type}", store_type_group=store_type)
            df_list.append(merged_df.merge(df, how="left", on=[geo_group, store_type])[['store_id', f"{geo_group}_{store_type}"]])
    
    dfs = [df.set_index('store_id') for df in df_list]
    return pd.concat(dfs, axis=1)

def stores_in_radius_by_type(stores_df, plaace_df, store_types, radius=0.1):
    df_list = []
    df_list.append(stores_in_radius(stores_df, plaace_df, radius=radius).rename(columns={'count':'number_of_all_stores'})) # All stores in radius
    
    for store_type in store_types:
        df = stores_in_radius(stores_df, plaace_df, store_type_group=store_type, radius=radius)
        df.rename(columns={'count': f'number_of_{store_type}'}, inplace=True)
        df_list.append(df)
    
    return pd.concat(df_list, axis=1)