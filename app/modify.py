import os
from datetime import datetime

# ... [your previous imports]

HISTORY_DIR = 'history'

# Ensure history directory exists
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

# ... [your previous code]

elif mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Compute statistics
        predictions = predict_fraud(data)
        total = len(predictions)
        fraudulent = sum(predictions >= 0.5)
        fraud_rate = (fraudulent / total) * 100

        # Display stats
        st.write(f"Total Transactions: {total}")
        st.write(f"Fraudulent Transactions: {fraudulent}")
        st.write(f"Fraud Rate: {fraud_rate:.2f}%")

        # Save the uploaded data for history with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data['Prediction'] = predictions
        data.to_csv(os.path.join(HISTORY_DIR, f"data_{timestamp}.csv"), index=False)

        # Visualize historical data
        history_files = sorted(os.listdir(HISTORY_DIR), reverse=True)
        selected_history = st.selectbox("Select a historical dataset", history_files)
        if selected_history:
            history_data = pd.read_csv(os.path.join(HISTORY_DIR, selected_history))
            fraud_data = history_data[history_data['Prediction'] >= 0.5]
            st.write(fraud_data)
