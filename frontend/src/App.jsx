import { useState, useEffect } from "react";
import "./styles.css";

const API_BASE_URL = "http://127.0.0.1:8000";

function App() {
  const [form, setForm] = useState({
    gender: "Male",
    number_of_dependents: 0,
    income: 50000,
    customer_tenure: 3,
    credit_score: 650,
    credit_history_length: 5,
    outstanding_loans: 10000,
    balance: 80000,
    num_of_products: 2,
    num_complaints: 0,
    age: 35,
    marital_status: "Married",
    education_level: "High School",
    customer_segment: "Retail",
    preferred_communication_channel: "Phone",
    age_group: "26-35",
    tenure_group: "1-2y",
    credit_category: "Good",
    occupation_encoded: "",
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [backendInfo, setBackendInfo] = useState(null);
  const [backendError, setBackendError] = useState("");

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/mlflow/config`);
        if (!res.ok) return;
        const data = await res.json();
        setBackendInfo(data);
      } catch {
        setBackendError("Backend not reachable");
      }
    };

    fetchConfig();
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]:
        name === "number_of_dependents" ||
        name === "income" ||
        name === "customer_tenure" ||
        name === "credit_score" ||
        name === "credit_history_length" ||
        name === "outstanding_loans" ||
        name === "balance" ||
        name === "num_of_products" ||
        name === "num_complaints" ||
        name === "age" ||
        name === "occupation_encoded"
          ? value === ""
            ? ""
            : Number(value)
          : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const payload = {
        ...form,
        occupation_encoded:
          form.occupation_encoded === "" ? null : Number(form.occupation_encoded),
      };

      const res = await fetch(`${API_BASE_URL}/predict_customer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Request failed");
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Bank Churn Prediction</h1>
        <p className="subtitle">
          Enter customer details and get a churn prediction from the MLflow-backed
          FastAPI model.
        </p>
      </header>

      <main className="app-main">
        <section className="card card--form">
          <h2>Customer details</h2>
          <form className="form" onSubmit={handleSubmit}>
            <div className="form-grid">
              <label>
                Gender
                <select
                  name="gender"
                  value={form.gender}
                  onChange={handleChange}
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </label>

              <label>
                Age
                <input
                  type="number"
                  name="age"
                  value={form.age}
                  onChange={handleChange}
                  min={18}
                  max={100}
                />
              </label>

              <label>
                Age group
                <select
                  name="age_group"
                  value={form.age_group}
                  onChange={handleChange}
                >
                  <option value="18-25">18-25</option>
                  <option value="26-35">26-35</option>
                  <option value="36-45">36-45</option>
                  <option value="46-55">46-55</option>
                  <option value="56-65">56-65</option>
                  <option value="65+">65+</option>
                </select>
              </label>

              <label>
                Number of dependents
                <input
                  type="number"
                  name="number_of_dependents"
                  value={form.number_of_dependents}
                  onChange={handleChange}
                  min={0}
                />
              </label>

              <label>
                Income
                <input
                  type="number"
                  name="income"
                  value={form.income}
                  onChange={handleChange}
                  min={0}
                />
              </label>

              <label>
                Customer tenure (years)
                <input
                  type="number"
                  name="customer_tenure"
                  value={form.customer_tenure}
                  onChange={handleChange}
                  min={0}
                  step="0.5"
                />
              </label>

              <label>
                Tenure group
                <select
                  name="tenure_group"
                  value={form.tenure_group}
                  onChange={handleChange}
                >
                  <option value="0-6m">0-6m</option>
                  <option value="6-12m">6-12m</option>
                  <option value="1-2y">1-2y</option>
                  <option value="2y+">2y+</option>
                </select>
              </label>

              <label>
                Credit score
                <input
                  type="number"
                  name="credit_score"
                  value={form.credit_score}
                  onChange={handleChange}
                  min={300}
                  max={900}
                />
              </label>

              <label>
                Credit score category
                <select
                  name="credit_category"
                  value={form.credit_category}
                  onChange={handleChange}
                >
                  <option value="Poor">Poor</option>
                  <option value="Fair">Fair</option>
                  <option value="Good">Good</option>
                  <option value="Very Good">Very Good</option>
                  <option value="Excellent">Excellent</option>
                </select>
              </label>

              <label>
                Credit history length (years)
                <input
                  type="number"
                  name="credit_history_length"
                  value={form.credit_history_length}
                  onChange={handleChange}
                  min={0}
                />
              </label>

              <label>
                Outstanding loans
                <input
                  type="number"
                  name="outstanding_loans"
                  value={form.outstanding_loans}
                  onChange={handleChange}
                  min={0}
                />
              </label>

              <label>
                Balance
                <input
                  type="number"
                  name="balance"
                  value={form.balance}
                  onChange={handleChange}
                  min={0}
                />
              </label>

              <label>
                Number of products
                <input
                  type="number"
                  name="num_of_products"
                  value={form.num_of_products}
                  onChange={handleChange}
                  min={1}
                />
              </label>

              <label>
                Number of complaints
                <input
                  type="number"
                  name="num_complaints"
                  value={form.num_complaints}
                  onChange={handleChange}
                  min={0}
                />
              </label>

              <label>
                Marital status
                <select
                  name="marital_status"
                  value={form.marital_status}
                  onChange={handleChange}
                >
                  <option value="Married">Married</option>
                  <option value="Single">Single</option>
                  <option value="Divorced">Divorced</option>
                  <option value="Widowed">Widowed</option>
                </select>
              </label>

              <label>
                Education level
                <select
                  name="education_level"
                  value={form.education_level}
                  onChange={handleChange}
                >
                  <option value="High School">High School</option>
                  <option value="Diploma">Diploma</option>
                  <option value="Master's">Master's</option>
                  <option value="Other">Other</option>
                </select>
              </label>

              <label>
                Customer segment
                <select
                  name="customer_segment"
                  value={form.customer_segment}
                  onChange={handleChange}
                >
                  <option value="Retail">Retail</option>
                  <option value="SME">SME</option>
                  <option value="Corporate">Corporate</option>
                </select>
              </label>

              <label>
                Preferred communication
                <select
                  name="preferred_communication_channel"
                  value={form.preferred_communication_channel}
                  onChange={handleChange}
                >
                  <option value="Phone">Phone</option>
                  <option value="Email">Email</option>
                  <option value="Branch">Branch</option>
                </select>
              </label>

              <label>
                Occupation (encoded, optional)
                <input
                  type="number"
                  name="occupation_encoded"
                  value={form.occupation_encoded}
                  onChange={handleChange}
                  min={0}
                  placeholder="Leave empty to ignore"
                />
              </label>
            </div>

            <button type="submit" className="primary-btn" disabled={loading}>
              {loading ? "Predicting..." : "Predict churn"}
            </button>
          </form>

          {error && <p className="error-text">{error}</p>}
        </section>

        <section className="card card--summary">
          <h2>Prediction summary</h2>
          <div className="summary-header">
            {backendInfo ? (
              <div className="pill">
                <span className="pill-dot" />
                <span className="pill-text">
                  {backendInfo.model_name || "MLflow model"} ·{" "}
                  {backendInfo.experiment_name || "churn_prediction"}
                </span>
              </div>
            ) : (
              <div className="pill pill--muted">
                <span className="pill-dot pill-dot--muted" />
                <span className="pill-text">
                  {backendError || "Connecting to MLflow backend..."}
                </span>
              </div>
            )}
          </div>

          {result ? (
            <>
              <div className="chips-row">
                <span
                  className={
                    result.prediction === 1
                      ? "badge badge--high"
                      : "badge badge--low"
                  }
                >
                  {result.prediction === 1 ? "High churn risk" : "Low churn risk"}
                </span>
              </div>

              <div className="probability-section">
                <div className="probability-label-row">
                  <span>Churn probability</span>
                  <span className="probability-value">
                    {(result.probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="probability-bar">
                  <div
                    className="probability-bar-fill"
                    style={{ width: `${Math.min(result.probability * 100, 100)}%` }}
                  />
                </div>
              </div>

              <p className="hint-text">
                Use this prediction together with customer history and business rules
                before making final retention decisions.
              </p>
            </>
          ) : (
            <div className="empty-state">
              <p>No prediction yet.</p>
              <p className="hint-text">
                Fill out the customer details on the left and click{" "}
                <strong>Predict churn</strong> to see the risk and probability here.
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
