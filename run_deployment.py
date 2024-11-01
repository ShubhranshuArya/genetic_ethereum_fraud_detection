import click
from constants.string_constants import StringConstants
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)


# This command is used to stop the deployment service.
@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
def run_main(stop_service: bool):
    """This function runs the fraud detection deployment pipeline.
    It also has an option to stop the prediction service when done.
    """
    model_name = StringConstants.name

    # If the stop-service flag is set, stop the prediction service.
    if stop_service:
        # Get the MLflow model deployer stack component
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()

        # Fetch existing services with same pipeline name, step name, and model name
        existing_services = model_deployer.find_model_server(
            pipeline_name=StringConstants.pipeline_name,
            pipeline_step_name=StringConstants.pipeline_step_name,
            model_name=model_name,
            running=True,
        )

        # If there are existing services, stop the first one.
        if existing_services:
            existing_services[0].stop(timeout=10)
        return

    # Run the continuous deployment pipeline
    continuous_deployment_pipeline()

    # Get the active model deployer
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # Run the inference pipeline
    inference_pipeline()

    # Print the command to inspect the experiment runs within the mlflow UI.
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs."
    )

    # Fetch existing services with the same pipeline name, step name, and model name
    service = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
    )

    # If there is a service running, print the prediction URL.
    if service[0]:
        print(
            f"The MLflow prediction server is running locally as a daemon "
            f"process and accepts inference requests at:\n"
            f"    {service[0].prediction_url}\n"
            f"To stop the service, re-run the same command and supply the "
            f"`--stop-service` argument."
        )


if __name__ == "__main__":
    run_main()