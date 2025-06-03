import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/VarunRd-124/mlops_project_water_potability.mlflow")

dagshub.init(repo_owner='VarunRd-124', repo_name='mlops_project_water_potability', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)