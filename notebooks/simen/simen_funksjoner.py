def average_revenue_of_chain(dataset_stores):
    "Average revenue of chains in datasett"
    dataset_stores = dataset_stores[(dataset_stores["year"] == 2016)]
    return dataset_stores.groupby(['chain_name'])['revenue'].mean()

def average_revenue_of_mall(dataset_stores):
    "Average revenue of malls in dataset"
    dataset_stores = dataset_stores[(dataset_stores["year"] == 2016)]
    return dataset_stores.groupby(['mall_name'])['revenue'].mean()

def population(dataset_age): 
    age_df = dataset_age[(dataset_age["year"] == 2016)]
    population = age_df.drop(["grunnkrets_id", "year"], axis=1).sum(axis=1)
    age_df["population_count"] = population 
    return age_df[["grunnkrets_id", "population_count"]]

def population_grouped(data_age, data_geography, grouping_element): 
    age_df = population(data_age)
    geography_df = data_geography[data_geography["year"] == 2016]
    population_df = age_df.merge(geography_df, how="left", on ="grunnkrets_id")
    grouped_df = population_df.groupby([grouping_element], as_index = False)["population_count"].sum()
    return grouped_df

def mean_income_per_capita(dataset_age,dataset_income):
    "mean income per capita per grunnkrets"
    age_df = population(data_age)
    income_df = dataset_income[dataset_income["year"] == 2016]
    age_and_income_df = age_df.merge(income_df, how='left', on='grunnkrets_id')
    mean_income = age_and_income_df.drop(['year','singles','couple_without_children','couple_with_children','other_households','single_parent_with_children'],axis=1)
    mean_income['mean_income']=mean_income['all_households']/mean_income['population_count']
    mean_income=mean_income.drop(['all_households'], axis=1)

    return mean_income

def mean_income_per_capita_grouped(dataset_income,dataset_geography,grouping_element):
    data_mean_income = mean_income_per_capita(data_age,dataset_income)
    geography_df = dataset_geography[dataset_geography["year"] == 2016]
    mean_income_geo_df = data_mean_income.merge(geography_df, how='left', on='grunnkrets_id')
    grouped_population_df=mean_income_geo_df.groupby([grouping_element], as_index = False)["population_count"].sum()
    grouped_income_df=mean_income_geo_df.groupby([grouping_element], as_index = False)["mean_income"].mean()
    mean_income_geo_df=grouped_income_df.merge(grouped_population_df, how='left', on=grouping_element)
    mean_income_geo_df['mean_income']=mean_income_geo_df['mean_income']/mean_income_geo_df['population_count']
    finished_mean = mean_income_geo_df.drop(['population_count'],axis=1)
    return finished_mean