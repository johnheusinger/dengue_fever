from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_model, make_prediction


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["scaled_values", "params:model_options"],
                outputs="regressor",
                name="train_model",
            ),
            node(
                func=make_prediction,
                inputs=["scaled_values", "regressor", "submission_format"],
                outputs="submission_data",
                name="make_prediction",
            )
        ]
    )
