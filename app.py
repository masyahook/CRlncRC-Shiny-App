from typing import List

from shiny import *
from shiny.types import NavSetArg

from ml_tools import run_ML_pipeline


def nav_controls() -> List[NavSetArg]:
    """Navigation tabs at the top of Shiny app"""
    return [
        ui.nav(

            "Machine Learning",  # The title of the the current navigation bar

            ui.panel_title('Run ML model training'),  # 

            ui.layout_sidebar(
                ui.panel_sidebar(
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
                        )
                    )
                )
            ),
        ),
        ui.nav("PCA",  ui.output_text("my_cool_text"),
                     ui.output_text_verbatim("a_code_block")),
        ui.nav('About')
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
        return run_ML_pipeline('classification_metrics', input.ml_model())

    @output
    @render.plot
    @reactive.event(input.run)
    async def confusion_matrix():
        return run_ML_pipeline('confusion_matrix', input.ml_model())

    @output
    @render.plot
    @reactive.event(input.run)
    async def roc_auc_curve():
        return run_ML_pipeline('roc_auc_curve', input.ml_model())

    @output
    @render.plot
    @reactive.event(input.run)
    async def feature_importance():
        return run_ML_pipeline('feature_importance', input.ml_model())


app = App(app_ui, server)