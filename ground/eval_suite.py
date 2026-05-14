import json
from typing import Dict, Any, Union

def evaluate_mission_intelligence(json_telemetry: Union[str, Dict[str, Any]], llm_report: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates the 'Mission Intelligence' metric for the OSP presentation.
    Calculates Faithfulness (Grounding Accuracy) by checking for Hallucinations vs. Ground Truth.
    """
    
    # Parse inputs if they are strings
    if isinstance(json_telemetry, str):
        try:
            json_telemetry = json.loads(json_telemetry)
        except json.JSONDecodeError:
            json_telemetry = {}
            
    if isinstance(llm_report, str):
        try:
            llm_report = json.loads(llm_report)
        except json.JSONDecodeError:
            # If it's pure text, we might try to extract count, but our pipeline usually returns JSON
            llm_report = {}

    # 1. Ground Truth from Telemetry
    ground_truth_anomalies = json_telemetry.get('anomalies', [])
    detected_in_json = len(ground_truth_anomalies)

    # 2. Extract reported entities from LLM
    # The llm_analyst.py outputs structured JSON with 'anomaly_assessments'
    if 'anomaly_assessments' in llm_report:
        reported_by_llm = len(llm_report['anomaly_assessments'])
    else:
        # Fallback: check if we can extract a count from summary or text
        # (This is a simplified check for the presentation)
        text_to_check = str(llm_report).lower()
        reported_by_llm = sum(1 for a in ground_truth_anomalies if a.get('type', '').lower() in text_to_check)
        if reported_by_llm == 0 and detected_in_json > 0:
            # Maybe it just mentioned 'ship' or 'harbor'
            reported_by_llm = text_to_check.count('ship') + text_to_check.count('harbor') # very basic fallback
            reported_by_llm = min(reported_by_llm, detected_in_json)

    # 3. Calculate Faithfulness Accuracy Metric
    # Accuracy is 1.0 if the LLM didn't hallucinate new entities or miss existing ones
    is_faithful = (detected_in_json == reported_by_llm)
    accuracy = 1.0 if is_faithful else 0.0

    # Determine failure mode for detailed reporting
    failure_mode = None
    if reported_by_llm > detected_in_json:
        failure_mode = "Hallucination (Over-reporting)"
    elif reported_by_llm < detected_in_json:
        failure_mode = "Omission (Under-reporting)"

    results = {
        "metric_name": "Grounding Accuracy (Faithfulness)",
        "accuracy": accuracy,
        "is_faithful": is_faithful,
        "ground_truth_count": detected_in_json,
        "llm_reported_count": reported_by_llm,
        "failure_mode": failure_mode,
        "details": f"Telemetry detected {detected_in_json} entities. LLM reported {reported_by_llm} entities."
    }

    return results

if __name__ == "__main__":
    # Test cases for the evaluation script
    
    # 1. Perfect Match (Faithful)
    mock_telemetry = {"anomalies": [{"type": "ship"}, {"type": "ship"}]}
    mock_llm_response = {"anomaly_assessments": [{"type": "ship"}, {"type": "ship"}]}
    print("Test 1: Perfect Match")
    print(json.dumps(evaluate_mission_intelligence(mock_telemetry, mock_llm_response), indent=2))
    
    # 2. Hallucination
    mock_telemetry_2 = {"anomalies": [{"type": "ship"}]}
    mock_llm_response_2 = {"anomaly_assessments": [{"type": "ship"}, {"type": "ship", "reasoning": "Hallucinated second ship"}]}
    print("\nTest 2: Hallucination")
    print(json.dumps(evaluate_mission_intelligence(mock_telemetry_2, mock_llm_response_2), indent=2))
    
    # 3. Omission
    mock_telemetry_3 = {"anomalies": [{"type": "ship"}, {"type": "harbor"}]}
    mock_llm_response_3 = {"anomaly_assessments": [{"type": "ship"}]}
    print("\nTest 3: Omission")
    print(json.dumps(evaluate_mission_intelligence(mock_telemetry_3, mock_llm_response_3), indent=2))
