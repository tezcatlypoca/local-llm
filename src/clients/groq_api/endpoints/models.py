import requests


def get_models(groq_client):
    url = f"{groq_client.api_url}/openai/v1/models"
    res = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {groq_client.api_key}",
            "Content-Type": "application/json"
        }
    ).json()

    formated_res = _format_models_response(res['data'])
    # print(len(formated_res))
    return formated_res

def _format_models_response(datas):
    model_data = []

    for data in datas:
        if data['object'] == 'model' and data['active']:
            model_data.append({
                "identifier": data['id'],
                "context_window": data['context_window'],
                "max_completion_tokens": data['max_completion_tokens'],
            })
    return model_data
