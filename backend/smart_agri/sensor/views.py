from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import SensorData
from .serializers import SensorDataSerializer


# 🔹 POST API (ESP32 sends data)
@api_view(['POST'])
def add_sensor_data(request):
    serializer = SensorDataSerializer(data=request.data)
    
    if serializer.is_valid():
        serializer.save()
        return Response({"message": "Data saved successfully"})
    
    return Response(serializer.errors)


# 🔹 GET API (Frontend fetch)
@api_view(['GET'])
def get_sensor_data(request):
    data = SensorData.objects.all().order_by('-created_at')[:20]
    serializer = SensorDataSerializer(data, many=True)
    return Response(serializer.data)