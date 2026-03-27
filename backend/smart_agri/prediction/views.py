from django.http import JsonResponse

def test_prediction(request):
    return JsonResponse({"message": "Prediction API working"})