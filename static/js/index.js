function generatePayload() {

    const object = {

        "age": document.getElementById('age').value,
        "height1": document.getElementById('height1').value,
        "height2": document.getElementById('height2').value,
        "weight": document.getElementById('weight').value,
        "sex": document.getElementById('sex').value,
        "race7": document.getElementById('race7').value,
        "educat": document.getElementById('educat').value,
        "marital": document.getElementById('marital').value,
        "occupat": document.getElementById('occupat').value,
        "cig_stat": document.getElementById('cig_stat').value,
        "cig_years": document.getElementById('cig_years').value,
        "cigpd_f": document.getElementById('cigpd_f').value,
        "cigar": document.getElementById('cigar').value,
        "pipe": document.getElementById('pipe').value,
        "fh_cancer": document.getElementById('fh_cancer').value,
        "colo_fh": document.getElementById('colo_fh').value,
        "colo_fh_cnt": document.getElementById('colo_fh_cnt').value,
        "asppd": document.getElementById('asppd').value,
        "ibuppd": document.getElementById('ibuppd').value,
        "arthrit_f": document.getElementById('arthrit_f').value,
        "bronchit_f": document.getElementById('bronchit_f').value,
        "colon_comorbidity": document.getElementById('colon_comorbidity').value,
        "diabetes_f": document.getElementById('diabetes_f').value,
        "divertic_f": document.getElementById('divertic_f').value,
        "emphys_f": document.getElementById('emphys_f').value,
        "gallblad_f": document.getElementById('gallblad_f').value,
        "hearta_f": document.getElementById('hearta_f').value,
        "hyperten_f": document.getElementById('hyperten_f').value,
        "liver_comorbidity": document.getElementById('liver_comorbidity').value,
        "osteopor_f": document.getElementById('osteopor_f').value,
        "polyps_f": document.getElementById('polyps_f').value,
        "stroke_f": document.getElementById('stroke_f').value,

    }

    return object;
}

function postRequest() {
    let payload = generatePayload();
    const values = Object.values(payload);

    for (let i = 0; i < values.length; i++) {
        if (values[i].length === 0) {
            alert('Answer all text fields');
            return;
        } 
    }
    
    payload = JSON.stringify(payload);
     
    let xhr = new XMLHttpRequest();
    xhr.open('POST', 'http://127.0.0.1:5000/api'); // deployment: https://www.cancerprediction.org/api
    xhr.setRequestHeader('Content-type', 'application/json');
    xhr.send(payload);

    xhr.onload = function() {

        if (xhr.status != 200) { 
            alert(`Error ${xhr.status}: ${xhr.statusText}`);
        } 
        
        else { 

            json = xhr.responseText;
            obj = JSON.parse(json);
            results = obj.results;
            cancerPrediction = results.prediction;
            predictionProbability = results.probability;

            if (cancerPrediction === "no") {
                result = 'Based on your features, the model is '+predictionProbability+' confident that you can safely forgo a screening for colon cancer. Please remember that this is not a diagnosis';
                document.getElementById('result').innerHTML = result;
            }

            else if (cancerPrediction === 'yes') {
                result = 'Based on your features, the model is '+predictionProbability+' confident that you should get screened for colon cancer. Please remember that this is not a diagnosis';
                document.getElementById('result').innerHTML = result;
            }

        }

    };

    xhr.onerror = function() {
        alert('Request failed');
    };

}