from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.privacy_budget import PureDPBudget
from tmlt.analytics.session import Session
from tmlt.analytics.protected_change import AddOneRow


# def create_tmlt_analytics_session(source_id, df):
#     return Session.from_dataframe(
#         privacy_budget=PureDPBudget(epsilon=float('inf')),
#         source_id=source_id,
#         dataframe=df,
#         protected_change=AddOneRow(),
#     )


def tumult_ana_count(source_id="synthetic_data"):

    # session = create_tmlt_analytics_session(source_id, df)

    count_query = QueryBuilder(source_id).count()
    return count_query

    # private_count = session.evaluate(
    #     count_query,
    #     privacy_budget=PureDPBudget(epsilon=epsilon)
    # )
    # return private_count


def tumult_ana_mean(source_id="synthetic_data"):
    average_query = QueryBuilder(source_id).average()
    return average_query


def tumult_ana_sum(source_id="synthetic_data"):
    sum_query = QueryBuilder(source_id).sum()
    return sum_query

def tumult_ana_variance(column, epsilon, source_id="synthetic_data"):
    variance_query = QueryBuilder(source_id).variance()
    return variance_query