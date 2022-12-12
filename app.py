from typing import List

from shiny import *
from shiny.types import NavSetArg

from ml_tools import run_ML_pipeline, run_PCA


def nav_controls() -> List[NavSetArg]:
    """Navigation tabs at the top of Shiny app"""
    return [
        ui.nav(

            "Machine Learning",  # The title of the the current navigation bar

            ui.panel_title('Run ML model training'), 

            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.input_file("features", "Upload the feature set (should be .csv/.tsv file, the first column corresponds to target set)", accept=[".csv", '.tsv'], multiple=False),
                    ui.input_file("pos_target", "Upload the positive target set (should be a list of sample IDs formatted as a column)", multiple=False),
                    ui.input_file("neg_target", "Upload the negative target set (should be a list of sample IDs formatted as a column)", multiple=False),
                    ui.input_select(
                        'ml_model', 
                        'Choose ML model to train', {
                            'NB': 'Naive Bayes Classifier',
                            'kNN': 'kNN classifier',
                            'LR': 'Logistic Regression',
                            'RF': 'Random Forest',
                            'SVM': 'SVM classifier'
                            }
                    ),
                    ui.input_action_button(
                    "run", "Run training"
                )
                ),
                ui.panel_main(
                    ui.navset_tab(
                        ui.nav(
                            "Classification metrics", 
                            ui.output_table(
                                'classification_metrics', 
                                )
                        ),
                        ui.nav(
                            "Confusion matrix", 
                            ui.output_plot(
                                'confusion_matrix', 
                                width='500px', 
                                height='500px'
                                )
                        ),
                        ui.nav(
                            "ROC AUC curve", 
                            ui.output_plot(
                                'roc_auc_curve', 
                                width='500px', 
                                height='500px'
                                )
                        ),
                        ui.nav(
                            'Feature importance',
                            ui.output_plot(
                                'feature_importance',
                                width='1000px', 
                                height='500px'
                            )
                        ),
                    )
                )
            )
        ),
        ui.nav(
            "PCA",  

            ui.panel_title('Principal Component Analysis'),

            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.input_action_button(
                    "run_pca", "Run analysis"
                    )
                ),
                ui.panel_main(
                    ui.div(
                        {"class": "card mb-3"},
                        ui.div(
                            {"class": "card-body"},
                            ui.h5({"class": "card-title mt-0"}, "Principal components"),
                            ui.output_plot("pca", width='800px', height='500px')
                        )
                    )
                )
            )
        ),
        ui.nav(
                            'About',
                            ui.h1("Running ML analysis based on input date"),
                            ui.p(
                """
                This Shiny app provides the ability to train the model on the fly for the user."""
            )
                            )
    ]


app_ui = ui.page_navbar(
    *nav_controls(),
    title="CRlncRC analysis",
    inverse=True,
    id="navbar_id"
)


def server(input: Inputs, output: Outputs, session: Session):
    @reactive.Effect
    def _():
        print("Current navbar page: ", input.navbar_id())

    @output
    @render.table
    @reactive.event(input.run)
    async def classification_metrics():

        if input.features() is None:
            return "Please upload the feature set (.csv or .tsv files)"
        if input.pos_target() is None:
            return "Please upload the positive target set"
        if input.neg_target() is None:
            return "Please upload the negative target set"

        return run_ML_pipeline('classification_metrics', input)

    @output
    @render.plot
    @reactive.event(input.run)
    async def confusion_matrix():

        if input.features() is None:
            return "Please upload the feature set (.csv or .tsv files)"
        if input.pos_target() is None:
            return "Please upload the positive target set"
        if input.neg_target() is None:
            return "Please upload the negative target set"

        return run_ML_pipeline('confusion_matrix', input)

    @output
    @render.plot
    @reactive.event(input.run)
    async def roc_auc_curve():

        if input.features() is None:
            return "Please upload the feature set (.csv or .tsv files)"
        if input.pos_target() is None:
            return "Please upload the positive target set"
        if input.neg_target() is None:
            return "Please upload the negative target set"

        return run_ML_pipeline('roc_auc_curve', input)

    @output
    @render.plot
    @reactive.event(input.run)
    async def feature_importance():

        if input.features() is None:
            return "Please upload the feature set (.csv or .tsv files)"
        if input.pos_target() is None:
            return "Please upload the positive target set"
        if input.neg_target() is None:
            return "Please upload the negative target set"

        return run_ML_pipeline('feature_importance', input)

    @output
    @render.plot
    @reactive.event(input.run_pca)
    async def pca():

        if input.features() is None:
            return "Please upload the feature set (.csv or .tsv files)"
        if input.pos_target() is None:
            return "Please upload the positive target set"
        if input.neg_target() is None:
            return "Please upload the negative target set"

        return run_PCA(input)


app = App(app_ui, server)