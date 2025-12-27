import pandas as pd
from zenml.client import Client

def test_model_inference():
    # 1. Connect to ZenML and get the last run of your pipeline
    client = Client()
    # Note: Ensure the name matches exactly what is in your @pipeline decorator
    pipeline = client.get_pipeline("continous_deployment_pipeline")
    last_run = pipeline.last_run
    
    print(f"Fetching artifacts from run: {last_run.name}")

    # 2. Load the trained model artifact
    # The step name 'model_train' matches the function name in your pipeline
    model_step = last_run.steps["model_train"]
    model = model_step.output.load()
    
    print(f"Model loaded successfully: {type(model)}")

    # 3. Create a sample inputs (Dummy Data)
    # This must match the columns your model expects!
    # Based on the Olist dataset usually used, here are common columns.
    # IF THIS FAILS, check the columns in your X_train using 'print(X_train.columns)' in your training step.
    sample_data = pd.DataFrame({
        "payment_sequential": [1],
        "payment_installments": [1],
        "payment_value": [99.99],
        "price": [89.99],
        "freight_value": [10.00],
        "product_name_lenght": [40],
        "product_description_lenght": [200],
        "product_photos_qty": [1],
        "product_weight_g": [500],
        "product_length_cm": [20],
        "product_height_cm": [10],
        "product_width_cm": [15]
    })

    # 4. Make a prediction
    print("\nRunning prediction on sample data...")
    try:
        prediction = model.predict(sample_data)
        print(f"✅ Prediction successful!")
        print(f"Predicted Review Score: {prediction[0]:.2f}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print("Tip: Check if 'sample_data' columns match exactly what your model was trained on.")

if __name__ == "__main__":
    test_model_inference()