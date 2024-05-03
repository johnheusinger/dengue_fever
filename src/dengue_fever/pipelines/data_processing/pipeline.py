from kedro.pipeline import Pipeline, node, pipeline

from .nodes import merge_features_and_label, encoding, dropping_columns, impute_missing_values, scale


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=merge_features_and_label,
                inputs=["dengue_features_train", "dengue_features_test", "dengue_labels_train"],
                outputs="merged_features_and_label",
                name="merge_features_and_label",
            ),
            node(
                func=encoding,
                inputs="merged_features_and_label",
                outputs="encoded_features_and_label",
                name="encoding",
            ),         
            node(
                func=dropping_columns,
                inputs=["encoded_features_and_label", "params:columns_to_drop"],
                outputs="dropped_features_and_label",
                name="dropping_columns",
            ),
            node(
                func=impute_missing_values,
                inputs=["dropped_features_and_label"],
                outputs="imputed_values",
                name="impute_missing_values",
            ),
            node(
                func=scale,
                inputs=["imputed_values"],
                outputs="scaled_values",
                name="scale",
            )
        ]
    )
