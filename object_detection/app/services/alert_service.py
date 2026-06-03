import json

class Alert:
    def __init__(self, alert_type:str, alert_confidence:int):
        alert = {
            "Type":"",
            "Confidence":0,
        }
        if(alert_type == "no person"):
            alert["Type"] = "No person detected"
            alert["Confidence"] = alert_confidence
        
        if(alert_type == "multiple person"):
            alert["Type"] = "More than one person detected"
            alert["Confidence"] = alert_confidence

        if(alert_type == "phone"):
            alert["Type"] = "Phone detected"
            alert['Confidence'] = alert_confidence
        
        if(alert_type == "laptop"):
            alert["Type"] = "Laptop detected"
            alert["Confidence"] = alert_confidence

        if(alert_type == "multiple laptop"):
            alert["Type"] = "Multiple laptops detected"
            alert["Confidence"] = alert_confidence
        
        json_string = json.dumps(alert, indent=4)
        print(json_string)
        return
    
    
