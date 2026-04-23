import json
from datetime import date, timedelta
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone

from .models import SensorData, GrowthPrediction, PlantImage, CropSession, DailyGrowthRecord
from .ml_utils import predict_growth, predict_disease, extract_greenness, DISEASE_TREATMENT


# ─── Ensure default active session ──────────────────────────────────────────
def get_active_session():
    session = CropSession.objects.filter(is_active=True).first()
    if not session:
        today = date.today()
        session = CropSession.objects.create(
            crop_name='Green Gram', field_acres=1.0, sowing_date=today
        )
    return session


# ─── Dashboard ───────────────────────────────────────────────────────────────
def dashboard(request):
    session        = get_active_session()
    latest_sensor  = SensorData.objects.first()
    latest_pred    = GrowthPrediction.objects.first()
    recent_images  = PlantImage.objects.filter(processed=True)[:6]
    growth_records = list(DailyGrowthRecord.objects.filter(session=session).order_by('day_number', 'slot')[:60])

    # Build actual vs predicted chart data
    chart_days, actual_heights, pred_heights, pred_yields = [], [], [], []
    seen = {}
    for r in growth_records:
        key = r.day_number
        if key not in seen:
            seen[key] = True
            chart_days.append(r.day_number)
            actual_heights.append(r.actual_height_cm)
            pred_heights.append(r.predicted_height_cm)
            pred_yields.append(r.predicted_yield_per_acre)

    sensor_history = list(
        SensorData.objects.values('timestamp','ph','temperature','humidity','soil_moisture','npk_nitrogen')[:20]
    )
    for s in sensor_history:
        s['timestamp'] = s['timestamp'].strftime('%H:%M')

    # Today's upload status
    today = date.today()
    morning_done = DailyGrowthRecord.objects.filter(session=session, date=today, slot='morning', image_uploaded=True).exists()
    evening_done = DailyGrowthRecord.objects.filter(session=session, date=today, slot='evening', image_uploaded=True).exists()

    return render(request, 'agriculture/dashboard.html', {
        'latest_sensor':     latest_sensor,
        'latest_prediction': latest_pred,
        'recent_images':     recent_images,
        'session':           session,
        'morning_done':      morning_done,
        'evening_done':      evening_done,
        'sensor_history_json': json.dumps(sensor_history[::-1]),
        'chart_days_json':   json.dumps(chart_days),
        'actual_heights_json': json.dumps(actual_heights),
        'pred_heights_json': json.dumps(pred_heights),
        'pred_yields_json':  json.dumps(pred_yields),
    })


# ─── API: Sensor Data from ESP32 ────────────────────────────────────────────
@csrf_exempt
@require_http_methods(["POST"])
def api_sensor_data(request):
    try:
        body   = json.loads(request.body)
        sensor = SensorData.objects.create(
            ph=body.get('ph'), npk_nitrogen=body.get('npk_n') or body.get('nitrogen'),
            npk_phosphorus=body.get('npk_p') or body.get('phosphorus'),
            npk_potassium=body.get('npk_k') or body.get('potassium'),
            temperature=body.get('temperature'), humidity=body.get('humidity'),
            soil_moisture=body.get('soil_moisture') or body.get('moisture'),
            device_id=body.get('device_id', 'ESP32_01'),
        )
        session = get_active_session()
        result  = predict_growth(
            sensor.ph or 6.5, sensor.npk_nitrogen or 60, sensor.npk_phosphorus or 30,
            sensor.npk_potassium or 50, sensor.temperature or 27,
            sensor.humidity or 65, sensor.soil_moisture or 45,
            day_number=session.days_since_sowing, field_acres=session.field_acres
        )
        GrowthPrediction.objects.create(
            sensor_data=sensor,
            predicted_yield_per_acre=result['predicted_yield_per_acre'],
            growth_stage=result['growth_stage'],
            health_score=result['health_score'],
            recommendation=result['recommendation_text'],
            model_accuracy=result['model_accuracy'],
        )
        # Auto create DailyGrowthRecord if no image uploaded today
        today = date.today()
        slot  = 'morning' if timezone.now().hour < 14 else 'evening'
        record, created = DailyGrowthRecord.objects.get_or_create(
            session=session, date=today, slot=slot,
            defaults={
                'day_number': session.days_since_sowing,
                'predicted_height_cm': result['predicted_height_cm'],
                'predicted_yield_per_acre': result['predicted_yield_per_acre'],
                'health_score': result['health_score'],
                'growth_stage': result['growth_stage'],
                'recommendation': result['recommendation_text'],
                'model_accuracy': result['model_accuracy'],
                'sensor_data': sensor,
            }
        )
        if not created and not record.image_uploaded:
            record.predicted_height_cm      = result['predicted_height_cm']
            record.predicted_yield_per_acre = result['predicted_yield_per_acre']
            record.health_score             = result['health_score']
            record.growth_stage             = result['growth_stage']
            record.recommendation           = result['recommendation_text']
            record.model_accuracy           = result['model_accuracy']
            record.sensor_data              = sensor
            record.save()

        return JsonResponse({
            'status': 'ok', 'sensor_id': sensor.id,
            'predicted_yield_per_acre': result['predicted_yield_per_acre'],
            'predicted_height_cm': result['predicted_height_cm'],
            'health_score': result['health_score'],
            'growth_stage': result['growth_stage'],
            'model_accuracy': result['model_accuracy'],
            'recommendations': result['recommendations'],
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


@require_http_methods(["GET"])
def api_get_data(request):
    sensor = SensorData.objects.first()
    pred   = GrowthPrediction.objects.first()
    if not sensor:
        return JsonResponse({'status': 'no_data'})
    history = list(SensorData.objects.values(
        'timestamp','ph','temperature','humidity','soil_moisture',
        'npk_nitrogen','npk_phosphorus','npk_potassium')[:20])
    for h in history:
        h['timestamp'] = h['timestamp'].strftime('%Y-%m-%d %H:%M')
    session = get_active_session()
    return JsonResponse({
        'status': 'ok',
        'latest_sensor': {
            'ph': sensor.ph, 'temperature': sensor.temperature, 'humidity': sensor.humidity,
            'soil_moisture': sensor.soil_moisture, 'npk_n': sensor.npk_nitrogen,
            'npk_p': sensor.npk_phosphorus, 'npk_k': sensor.npk_potassium,
            'device_id': sensor.device_id,
            'timestamp': sensor.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'latest_prediction': {
            'predicted_yield_per_acre': pred.predicted_yield_per_acre if pred else None,
            'health_score': pred.health_score if pred else None,
            'growth_stage': pred.growth_stage if pred else None,
            'recommendation': pred.recommendation if pred else None,
            'model_accuracy': pred.model_accuracy if pred else None,
        } if pred else None,
        'days_since_sowing': session.days_since_sowing,
        'history': history[::-1],
    })


@require_http_methods(["GET"])
def api_predict_growth(request):
    try:
        session = get_active_session()
        result  = predict_growth(
            float(request.GET.get('ph', 6.5)),
            float(request.GET.get('n', 60)),
            float(request.GET.get('p', 30)),
            float(request.GET.get('k', 50)),
            float(request.GET.get('temp', 27)),
            float(request.GET.get('humidity', 65)),
            float(request.GET.get('moisture', 45)),
            day_number=session.days_since_sowing,
            field_acres=session.field_acres,
        )
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


# ─── API: Growth Chart Data ──────────────────────────────────────────────────
@require_http_methods(["GET"])
def api_growth_chart(request):
    import math
    session = get_active_session()
    records = DailyGrowthRecord.objects.filter(session=session).order_by('day_number', 'slot')
    seen, days, actual, predicted, yields = {}, [], [], [], []
    for r in records:
        if r.day_number not in seen:
            seen[r.day_number] = True
            days.append(r.day_number)
            actual.append(r.actual_height_cm)
            predicted.append(r.predicted_height_cm)
            yields.append(r.predicted_yield_per_acre)

    # Always include the theoretical ideal growth curve for green gram (55-day crop)
    # so chart is never empty — shown as "Expected Ideal" baseline
    TOTAL_DAYS = 60  # green gram harvest ~55-65 days
    ideal_days, ideal_heights, ideal_yields = [], [], []
    for d in range(1, TOTAL_DAYS + 1):
        # Green gram sigmoid height curve: max ~50cm
        h = round(5 + 45 * (1 / (1 + math.exp(-0.12 * (d - 28)))), 1)
        # Green gram yield ramp: ~400-600 kg/acre
        y = round(200 + 350 * (d / TOTAL_DAYS) ** 1.5, 1)
        ideal_days.append(d)
        ideal_heights.append(h)
        ideal_yields.append(y)

    return JsonResponse({
        'days': days, 'actual_heights': actual,
        'predicted_heights': predicted, 'predicted_yields': yields,
        'ideal_days': ideal_days, 'ideal_heights': ideal_heights,
        'ideal_yields': ideal_yields,
        'crop': session.crop_name, 'field_acres': session.field_acres,
        'total_days': TOTAL_DAYS,
    })


# ─── API: Disease Prediction ─────────────────────────────────────────────────
@csrf_exempt
@require_http_methods(["POST"])
def api_predict_disease(request):
    try:
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'error': 'No image provided'}, status=400)

        session = get_active_session()
        slot    = request.POST.get('slot', 'morning')

        plant_img = PlantImage.objects.create(
            image=image_file,
            farmer_name=request.POST.get('farmer_name', ''),
            location=request.POST.get('location', ''),
            slot=slot,
        )

        diseased, disease_name, confidence, model_acc, treatment = predict_disease(plant_img.image.path)
        greenness = extract_greenness(plant_img.image.path)

        plant_img.disease_detected = diseased
        plant_img.disease_name     = disease_name
        plant_img.confidence       = confidence
        plant_img.model_accuracy   = model_acc
        plant_img.processed        = True
        plant_img.greenness_score  = greenness
        plant_img.save()

        # Update / create today's DailyGrowthRecord
        today = date.today()
        record, _ = DailyGrowthRecord.objects.get_or_create(
            session=session, date=today, slot=slot,
            defaults={'day_number': session.days_since_sowing}
        )
        record.image             = plant_img
        record.image_uploaded    = True
        record.actual_greenness_score = greenness
        # Estimate actual height from greenness + day (simple heuristic when no ruler)
        if record.day_number:
            record.actual_height_cm = round(
                10 + 70 * (1 / (1 + 2.718 ** (-0.05 * (record.day_number - 60)))) * (greenness / 100) * 1.3, 1
            )
        record.save()

        return JsonResponse({
            'status': 'ok', 'image_id': plant_img.id,
            'disease_detected': diseased, 'disease_name': disease_name,
            'confidence': confidence, 'model_accuracy': model_acc,
            'greenness_score': greenness,
            'image_url': plant_img.image.url,
            'treatment': treatment,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def api_recent_images(request):
    images = PlantImage.objects.filter(processed=True)[:10]
    return JsonResponse({'images': [{
        'id': img.id, 'url': img.image.url,
        'disease_detected': img.disease_detected, 'disease_name': img.disease_name,
        'confidence': img.confidence, 'model_accuracy': img.model_accuracy,
        'greenness_score': img.greenness_score,
        'uploaded_at': img.uploaded_at.strftime('%Y-%m-%d %H:%M'),
        'farmer': img.farmer_name, 'slot': img.slot,
    } for img in images]})


# ─── API: Session Management ─────────────────────────────────────────────────
@csrf_exempt
@require_http_methods(["POST"])
def api_new_session(request):
    try:
        body = json.loads(request.body)
        CropSession.objects.filter(is_active=True).update(is_active=False)
        session = CropSession.objects.create(
            crop_name=body.get('crop_name', 'Green Gram'),
            field_acres=float(body.get('field_acres', 1.0)),
            sowing_date=body.get('sowing_date', date.today().isoformat()),
        )
        return JsonResponse({'status': 'ok', 'session_id': session.id, 'crop': session.crop_name})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
