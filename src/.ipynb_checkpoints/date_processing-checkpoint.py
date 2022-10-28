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
# This function calculates the population for each grunnkrets
# Returns a df with grunnkretsID in the first column and population_count in the second column


def population(dataset_age):
    age_df = dataset_age[(dataset_age["year"] == 2016)]
    population = age_df.drop(["grunnkrets_id", "year"], axis=1).sum(axis=1)
    age_df["population_count"] = population
    return age_df[["grunnkrets_id", "population_count"]]

# This function calculates the population in a district or municipality, by setting grouping_elemnt either to the district_name or municipality_name


def population_grouped(data_age, data_geography, grouping_element):
    age_df = population(data_age)
    geography_df = data_geography[data_geography["year"] == 2016]
    population_df = age_df.merge(geography_df, how="left", on="grunnkrets_id")
    grouped_df = population_df.groupby([grouping_element], as_index=False)[
        "population_count"].sum()
    return grouped_df

# This function calculates the density (population/area_km2) for the chosen grouping_element


def population_density(age_df, geo_df, grouping_element):
    age_data = population(age_df)
    geo_df = geo_df[geo_df["year"] == 2016]
    combined_df = age_data.merge(geo_df, how="left", on="grunnkrets_id")
    density_df = combined_df.groupby([grouping_element], as_index=False)[
        ["population_count", "area_km2"]].sum()
    density_df["Density"] = density_df["population_count"] / \
        density_df["area_km2"]
    return density_df

# This function checks wether or not a store is part of a mall or not


def is_mall(stores_df):
    df = stores_df.copy()
    df["is_mall"] = df["mall_name"].notna()
    return df[["store_id", "mall_name", "is_mall"]]

# This function checks wether or not a store is part of a chain or not


def is_chain(stores_df):
    df = stores_df.copy()
    df["is_chain"] = df["chain_name"].notna()
    return df[["store_id", "chain_name", "is_chain"]]

# This function calculates the population count per number of stores in a geographic region


def population_per_store(age_df, geo_df, stores_df, grouping_element):
    new_geo_df = geo_df[geo_df["year"] == 2016]
    pop_gk = population(age_df)
    pop_df = population_grouped(age_df, geo_df, grouping_element)
    combined_df = pop_gk.merge(stores_df, how="left", on="grunnkrets_id").merge(
        new_geo_df, how="left", on="grunnkrets_id")
    grouped_df = combined_df.groupby([grouping_element], as_index=False)[
        "store_id"].count()
    pop_per_store_df = grouped_df.merge(
        pop_df, how="inner", on=grouping_element)
    pop_per_store_df["population_per_num_stores"] = pop_per_store_df["population_count"] / \
        pop_per_store_df["store_id"]
    pop_per_store_df.rename(columns={"store_id": "num_stores"}, inplace=True)
    new_pop_per_store_df = pop_per_store_df.replace([np.inf, -np.inf], 0)
    return new_pop_per_store_df

# This function do the same as population_per_store but can also filter on store types


def population_per_store_types(stores_df, plaace_hierarchy, grunnkrets_df, age_df, agg_name, geo_group, store_type_group):
    grunnkrets_df_2016 = grunnkrets_df[grunnkrets_df["year"] == 2016]
    num_stores_types_by_geo_group = store_types_count_by_geo_group(
        stores_df, plaace_hierarchy, grunnkrets_df_2016, agg_name, geo_group, store_type_group)
    pop_grouped_by_geo = population_grouped(
        age_df, grunnkrets_df_2016, geo_group)
    combined_df = num_stores_types_by_geo_group.merge(
        pop_grouped_by_geo, how="left", on=geo_group)
    combined_df["population_per_num_store"] = combined_df["population_count"] / \
        combined_df[agg_name]
    return combined_df

# This function groups the age distrubution (0-90) into 7 buckets with and returns a table which represents the presentages each of these
# buckets corresponds to compared with the total amount of people living in the given geographic region s


def age_distrubution(grunnkrets_age_df, geographic_df, grouping_element):
    age_df = grunnkrets_age_df[grunnkrets_age_df["year"] == 2016]
    age_df1 = age_df.drop(["year"], axis=1)
    age_df1["num_kids"] = age_df1.iloc[:, 1:8].sum(axis=1)
    age_df1["num_kids+"] = age_df1.iloc[:, 8:14].sum(axis=1)
    age_df1["num_youths"] = age_df1.iloc[:, 14: 19].sum(axis=1)
    age_df1["num_youthAdult"] = age_df1.iloc[:, 19:27].sum(axis=1)
    age_df1["num_adult"] = age_df1.iloc[:, 27:37].sum(axis=1)
    age_df1["num_adults+"] = age_df1.iloc[:, 37:62].sum(axis=1)
    age_df1["num_pensinors"] = age_df1.iloc[:, 62:92].sum(axis=1)

    age_df2 = age_df1[["grunnkrets_id", "num_kids", "num_kids+", "num_youths",
                       "num_youthAdult", "num_adult", "num_adults+", "num_pensinors"]]

    pop_df = population(grunnkrets_age_df)
    geo_df = geographic_df[geographic_df["year"] == 2016]
    new_geo_df = geo_df.drop(["geometry", "area_km2", "year"], axis=1)
    combined_df = age_df2.merge(pop_df, how="inner", on="grunnkrets_id").merge(
        new_geo_df, how="inner", on="grunnkrets_id")
    list_columns = ["num_kids", "num_kids+", "num_youths",
                    "num_youthAdult", "num_adult", "num_adults+", "num_pensinors"]
    combined_df2 = combined_df.groupby([grouping_element], as_index=False)[
        list_columns].sum()

    pop_gk = population_grouped(
        grunnkrets_age_df, geographic_df, grouping_element)
    new_df = combined_df2.merge(pop_gk, how="inner", on=grouping_element)

    new_df["kids_%"] = new_df["num_kids"] / new_df["population_count"]
    new_df["kids+_%"] = new_df["num_kids+"] / new_df["population_count"]
    new_df["youths_%"] = new_df["num_youths"] / new_df["population_count"]
    new_df["youthAdult_%"] = new_df["num_youthAdult"] / \
        new_df["population_count"]
    new_df["adult_%"] = new_df["num_adult"] / new_df["population_count"]
    new_df["adults+_%"] = new_df["num_adults+"] / new_df["population_count"]
    new_df["pensinors_%"] = new_df["num_pensinors"] / \
        new_df["population_count"]

    age_dist_df = new_df.drop(["population_count"], axis=1)
    # if (grouping_element == "grunnkrets_id"):
    # return new_df[["grunnkrets_id", "kids_%", "kids+_%", "youths_%", "youthAdult_%", "adult_%", "adults+_%", "pensinors_%" ]]
    # else:
    # return new_df[[grouping_element, "kids_%", "kids+_%", "youths_%", "youthAdult_%", "adult_%", "adults+_%", "pensinors_%" ]]

    return age_dist_df

# This function calculates the total amount of household types based on a geographic area


def household_type_distrubution(geographic_df, household_df, grouping_element):
    house_df = household_df[household_df["year"] == 2016]
    geo_df = geographic_df[geographic_df["year"] == 2016]
    combined_df = geo_df.merge(house_df, how="inner", on="grunnkrets_id")

    list_columns = ["couple_children_0_to_5_years", "couple_children_18_or_above", "couple_children_6_to_17_years",
                    "couple_without_children", "single_parent_children_0_to_5_years", "single_parent_children_18_or_above",
                    "single_parent_children_6_to_17_years", "singles"]

    grouped_df = combined_df.groupby([grouping_element], as_index=False)[
        list_columns].sum()
    grouped_df["tot_pop_count"] = grouped_df.iloc[:, 1:].sum(axis=1)

    grouped_df["%_dist_of_couple_children_0_to_5_years"] = grouped_df["couple_children_0_to_5_years"] / \
        grouped_df["tot_pop_count"]
    grouped_df["%_dist_of_couple_children_18_or_above"] = grouped_df["couple_children_18_or_above"] / \
        grouped_df["tot_pop_count"]
    grouped_df["%_dist_of_couple_children_6_to_17_years"] = grouped_df["couple_children_6_to_17_years"] / \
        grouped_df["tot_pop_count"]
    grouped_df["%_dist_of_couple_without_children"] = grouped_df["couple_without_children"] / \
        grouped_df["tot_pop_count"]
    grouped_df["%_dist_of_single_parent_children_0_to_5_years"] = grouped_df["single_parent_children_0_to_5_years"] / \
        grouped_df["tot_pop_count"]
    grouped_df["%_dist_of_single_parent_children_18_or_above"] = grouped_df["single_parent_children_18_or_above"] / \
        grouped_df["tot_pop_count"]
    grouped_df["%_dist_of_single_parent_children_6_to_17_years"] = grouped_df["single_parent_children_6_to_17_years"] / \
        grouped_df["tot_pop_count"]
    grouped_df["%_dist_of_singles"] = grouped_df["singles"] / \
        grouped_df["tot_pop_count"]

    returned_df = grouped_df.drop(["tot_pop_count"], axis=1)
    return returned_df


# Simens functions
def average_revenue_of_chain(dataset_stores):
    "Average revenue of chains in datasett"
    dataset_stores = dataset_stores[(dataset_stores["year"] == 2016)]
    return dataset_stores.groupby(['chain_name'])['revenue'].mean()


def average_revenue_of_mall(dataset_stores):
    "Average revenue of malls in dataset"
    dataset_stores = dataset_stores[(dataset_stores["year"] == 2016)]
    return dataset_stores.groupby(['mall_name'])['revenue'].mean()


def mean_income_per_capita(dataset_age, dataset_income):
    "mean income per capita per grunnkrets"
    age_df = population(dataset_age)
    income_df = dataset_income[dataset_income["year"] == 2016]
    age_and_income_df = age_df.merge(income_df, how='left', on='grunnkrets_id')
    mean_income = age_and_income_df.drop(['year', 'singles', 'couple_without_children',
                                         'couple_with_children', 'other_households', 'single_parent_with_children'], axis=1)
    mean_income['mean_income'] = mean_income['all_households'] / \
        mean_income['population_count']
    mean_income = mean_income.drop(['all_households'], axis=1)

    return mean_income


def mean_income_per_capita_grouped(dataset_age, dataset_income, dataset_geography, grouping_element):
    # gets data from mean_income_per_capita functino
    data_mean_income = mean_income_per_capita(dataset_age, dataset_income)
    # gets data from geography set and makes sure we only use data for 2016
    geography_df = dataset_geography[dataset_geography["year"] == 2016]
    # gets the data of mean income with the geography data
    mean_income_geo_df = data_mean_income.merge(
        geography_df, how='left', on='grunnkrets_id')
    # sum the number of people based on grouping element
    grouped_population_df = mean_income_geo_df.groupby(
        [grouping_element], as_index=False)["population_count"].sum()
    # merge this with the grunnkrets to see both total population per selected area and grunnkrets
    total_grouped_df = mean_income_geo_df.merge(
        grouped_population_df, how='left', on=grouping_element)
    portion_income_df = total_grouped_df
    # find ration of grunnkrets to total population and multiply this with grunnkrets mean income
    portion_income_df['mean_income'] = total_grouped_df['mean_income'] * \
        total_grouped_df['population_count_x'] / \
        total_grouped_df['population_count_y']
    # add these incomes together, should add up to the total mean income for the selected area
    grouped_income_df = portion_income_df.groupby(
        [grouping_element], as_index=False)["mean_income"].sum()
    return grouped_income_df

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

        df = pd.DataFrame.from_dict(
            store_count, orient='index', columns=['count'])
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
    merged_df = stores_df.merge(grunnkrets_df, how="left", on="grunnkrets_id").merge(
        plaace_df, how="left", on="plaace_hierarchy_id")

    df_list = []
    for geo_group in geo_groups:
        for store_type in store_types:
            df = store_types_count_by_geo_group(
                stores_df, plaace_df, grunnkrets_df, geo_group=geo_group, agg_name=f"{geo_group}_{store_type}", store_type_group=store_type)
            df_list.append(merged_df.merge(df, how="left", on=[geo_group, store_type])[
                           ['store_id', f"{geo_group}_{store_type}"]])

    dfs = [df.set_index('store_id') for df in df_list]
    return pd.concat(dfs, axis=1)


def store_types_all_revenue_by_geo_groups(stores_df, plaace_df, grunnkrets_df, store_types, geo_groups):
    merged_df = stores_df.merge(grunnkrets_df, how="left", on="grunnkrets_id").merge(
        plaace_df, how="left", on="plaace_hierarchy_id")

    df_list = []
    for geo_group in geo_groups:
        for store_type in store_types:
            df = store_types_revenue_by_geo_group(
                stores_df, plaace_df, grunnkrets_df, geo_group=geo_group, agg_name=f"{geo_group}_{store_type}", store_type_group=store_type)
            df_list.append(merged_df.merge(df, how="left", on=[geo_group, store_type])[
                           ['store_id', f"{geo_group}_{store_type}"]])

    dfs = [df.set_index('store_id') for df in df_list]
    return pd.concat(dfs, axis=1)


def stores_in_radius_by_type(stores_df, plaace_df, store_types, radius=0.1):
    df_list = []
    df_list.append(stores_in_radius(stores_df, plaace_df, radius=radius).rename(
        columns={'count': 'number_of_all_stores'}))  # All stores in radius

    for store_type in store_types:
        df = stores_in_radius(stores_df, plaace_df,
                              store_type_group=store_type, radius=radius)
        df.rename(columns={'count': f'number_of_{store_type}'}, inplace=True)
        df_list.append(df)

    return pd.concat(df_list, axis=1)
