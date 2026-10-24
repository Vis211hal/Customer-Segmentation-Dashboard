
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

artifacts = joblib.load("customer_segmentation_model.pkl")
features = artifacts["features"]
scaler = artifacts["scaler"]
kmeans = artifacts["kmeans"]

# Summary from your earlier training results.
# These are training-dataset statistics, not live customer data.
SEGMENT_STATS = {
    0: {
        "name": "Higher-Value Customers",
        "count": 1830,
        "avg_spent": 4330.30,
        "avg_orders": 7.77,
        "avg_order_value": 649.08,
        "description": "Higher average spending and purchase frequency."
    },
    1: {
        "name": "Lower-Value Customers",
        "count": 2509,
        "avg_spent": 383.72,
        "avg_orders": 1.72,
        "avg_order_value": 248.64,
        "description": "Lower average spending and fewer purchases."
    }
}

# In-memory history: resets when the Flask server restarts.
prediction_history = []


def dashboard_context(**extra):
    total_customers = sum(segment["count"] for segment in SEGMENT_STATS.values())
    context = {
        "segments": SEGMENT_STATS,
        "history": prediction_history,
        "total_customers": total_customers,
        "total_predictions": len(prediction_history),
        "segment_share": {
            segment_id: segment["count"] / total_customers * 100
            for segment_id, segment in SEGMENT_STATS.items()
        },
        "average_spend_gap": (
            SEGMENT_STATS[0]["avg_spent"] / SEGMENT_STATS[1]["avg_spent"]
        ),
    }
    context.update(extra)
    return context


@app.route("/")
def home():
    return render_template("index.html", **dashboard_context())


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [
            float(request.form["total_spent"]),
            float(request.form["total_quantity"]),
            float(request.form["total_transactions"]),
            float(request.form["average_order_value"])
        ]

        if not all(np.isfinite(values)):
            raise ValueError("Please enter valid finite numbers.")

        if any(value < 0 for value in values):
            raise ValueError("Values cannot be negative.")

        if values[2] <= 0:
            raise ValueError("Total transactions must be greater than zero.")

        customer_features = pd.DataFrame([values], columns=features)
        customer_features = np.log1p(customer_features)
        scaled_features = scaler.transform(customer_features)

        cluster = int(kmeans.predict(scaled_features)[0])
        segment = SEGMENT_STATS[cluster]

        prediction_history.insert(0, {
            "spent": values[0],
            "quantity": values[1],
            "transactions": values[2],
            "average_order_value": values[3],
            "cluster": cluster,
            "name": segment["name"]
        })
        del prediction_history[10:]

        return render_template("index.html", **dashboard_context(
            prediction=segment,
            cluster=cluster,
            form_values={
                "total_spent": values[0],
                "total_quantity": values[1],
                "total_transactions": values[2],
                "average_order_value": values[3]
            }
        ))

    except (ValueError, TypeError, KeyError) as error:
        return render_template("index.html", **dashboard_context(
            error=str(error),
            form_values={key: request.form.get(key, "") for key in (
                "total_spent", "total_quantity", "total_transactions", "average_order_value"
            )}
        ))


if __name__ == "__main__":
    app.run(debug=True)
