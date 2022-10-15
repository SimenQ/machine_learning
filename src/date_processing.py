"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

"""
UTILITY FUNCTIONS
"""
def lat_lon_bus_stop(bus_stops_df):
    bus_stops_df['lng_lat'] = bus_stops_df['geometry'].str.extract(r'\((.*?)\)')
    bus_stops_df[['lon', 'lat']] = bus_stops_df['lng_lat'].str.split(" ", 1, expand=True)
    bus_stops_df[['lon', 'lat']] = bus_stops_df[['lon', 'lat']].apply(pd.to_numeric)
    return bus_stops_df[['busstop_id', 'stopplace_type', 'importance_level', 'side_placement', 'geometry', 'lat', 'lon']]


def store_type_lookup(stores_df, plaace_df, store_ids, match):
    combined_df = stores_df.merge(plaace_df, how="inner", on="plaace_hierarchy_id")
    pass
"""
TRANSFORMS
"""

def store_type_in_dataaset(stores_df, plaace_df, lv_desc="lv1_desc"):
    combined_df = stores_df.merge(plaace_df, how="inner", on="plaace_hierarchy_id")
    return combined_df[lv_desc].value_counts().rename_axis(lv_desc).reset_index(name='count')


def stores_per_location_by_type(stores_df, plaace_df, grunnkrets_df, geo="district_name", lv_desc="lv1_desc"):
    """
    Number of stores of the same type in a geographic location.
    """
    combined_df = stores_df.merge(plaace_df, how="inner", on="plaace_hierarchy_id").merge(grunnkrets_df, how="inner", on="grunnkrets_id")
    return combined_df.groupby(by=[geo, lv_desc])['store_id'].count().reset_index(name='count')

def stores_revenue_per_location_by_type(stores_df, plaace_df, grunnkrets_df, geo="district_name", lv_desc="lv1_desc"):
    """
    Total revenue of stores of the same type in a geographic location.
    """
    combined_df = stores_df.merge(plaace_df, how="inner", on="plaace_hierarchy_id").merge(grunnkrets_df, how="inner", on="grunnkrets_id")
    return combined_df.groupby(by=[geo, lv_desc])['revenue'].sum().reset_index(name='total_revenue')

def stores_density_per_location_by_type(stores_df, plaace_df, grunnkrets_df, geo="district_name", lv_desc="lv1_desc"):
    """
    Density of stores of the same type in a geographic location.
    
    This depends on population
    """
    number_of_stores = stores_per_location_by_type(stores_df, plaace_df, grunnkrets_df, geo=geo, lv_desc=lv_desc)['count']
    population = 0
    return number_of_stores / population

def stores_in_radius(stores_df, plaace_df, radius=0.1, by_type=False, category=None):
    """
    Number of stores within a given radius. Can also indicate category to filter.
    """
    mat = cdist(stores_df[['lat', 'lon']], stores_df[['lat', 'lon']], metric='euclidean')
    new_df = pd.DataFrame(mat, index=stores_df['store_id'], columns=stores_df['store_id'])
    
    if by_type == False:
        count = pd.DataFrame(new_df[(new_df < radius) & (new_df > 0)].count(axis=1)).reset_index()
        count.rename(columns={0:'count'}, inplace=True)
        return count
    
    else:
        combined_df = stores_df.merge(plaace_df, how="inner", on="plaace_hierarchy_id")
        test_df = new_df[(new_df < 0.2) & (new_df > 0)]
        store_count = {}
        
        for index, row in test_df.iterrows():
            nearby_stores = row.dropna().index.values
            index_type = combined_df[combined_df['store_id'] == index][category].values[0]
            number_same = combined_df[(combined_df['store_id'].isin(nearby_stores)) & (combined_df[category] == index_type)]['store_id'].count()
            store_count[index] = number_same
            
        return pd.DataFrame.from_dict(store_count, orient='index', columns=['count']).reset_index()



def closest_bus_stop_cat(stores_df, bus_stops_df, cat="Regionalt knutepunkt"):
    """
    Id and distance of the closest bus stop to all stores.
    """
    bus_stops_df = bus_stops_df[bus_stops_df['importance_level'] == cat]
    mat = cdist(stores_df[['lat', 'lon']], bus_stops_df[['lat', 'lon']], metric='euclidean')

    new_df = pd.DataFrame(mat, index=stores_df['store_id'], columns=bus_stops_df['busstop_id'])
    
    stores = stores_df.store_id
    closest = new_df.idxmin(axis = 1)
    distance = new_df.min(axis = 1)
    
    return pd.DataFrame({'store_id' : stores.values, 'closest_bus_stop' : closest.values, 'distance' : distance.values})

def bus_stops_in_radius(stores_df, bus_stops_df, radius=0.1, cat=None):
    """
    Number of bus stops within a given radius. The importance level of bus stops can be specified.
    """
    if cat is not None:
        bus_stops_df = bus_stops_df[bus_stops_df['importance_level'] == cat]
        
    mat = cdist(stores_df[['lat', 'lon']], bus_stops_df[['lat', 'lon']], metric='euclidean')
    new_df = pd.DataFrame(mat, index=stores_df['store_id'], columns=bus_stops_df['busstop_id'])
    count = pd.DataFrame(new_df[new_df < radius ].count(axis=1)).reset_index()
    count.rename(columns={0:'count'}, inplace=True)
    return count
