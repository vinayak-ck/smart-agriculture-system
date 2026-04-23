from django.db import models


class SensorData(models.Model):
    timestamp       = models.DateTimeField(auto_now_add=True)
    ph              = models.FloatField(null=True, blank=True)
    npk_nitrogen    = models.FloatField(null=True, blank=True)
    npk_phosphorus  = models.FloatField(null=True, blank=True)
    npk_potassium   = models.FloatField(null=True, blank=True)
    temperature     = models.FloatField(null=True, blank=True)
    humidity        = models.FloatField(null=True, blank=True)
    soil_moisture   = models.FloatField(null=True, blank=True)
    device_id       = models.CharField(max_length=50, default='ESP32_01')

    class Meta:
        ordering = ['-timestamp']


class CropSession(models.Model):
    crop_name        = models.CharField(max_length=100, default='Rice')
    field_acres      = models.FloatField(default=1.0)
    sowing_date      = models.DateField()
    expected_harvest = models.DateField(null=True, blank=True)
    is_active        = models.BooleanField(default=True)
    created_at       = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.crop_name} ({self.sowing_date})"

    @property
    def days_since_sowing(self):
        from django.utils import timezone
        return (timezone.now().date() - self.sowing_date).days


class PlantImage(models.Model):
    image            = models.ImageField(upload_to='plant_images/')
    uploaded_at      = models.DateTimeField(auto_now_add=True)
    farmer_name      = models.CharField(max_length=100, blank=True)
    location         = models.CharField(max_length=100, blank=True)
    slot             = models.CharField(max_length=10, default='morning')
    disease_detected = models.BooleanField(null=True)
    disease_name     = models.CharField(max_length=100, blank=True)
    confidence       = models.FloatField(null=True, blank=True)
    model_accuracy   = models.FloatField(null=True, blank=True)
    processed        = models.BooleanField(default=False)
    greenness_score  = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['-uploaded_at']


class DailyGrowthRecord(models.Model):
    SLOT_CHOICES = [('morning','Morning'),('evening','Evening'),('auto','Auto (IoT only)')]

    session                 = models.ForeignKey(CropSession, on_delete=models.CASCADE, related_name='growth_records')
    date                    = models.DateField()
    slot                    = models.CharField(max_length=10, choices=SLOT_CHOICES, default='morning')
    day_number              = models.IntegerField(default=1)
    actual_height_cm        = models.FloatField(null=True, blank=True)
    actual_greenness_score  = models.FloatField(null=True, blank=True)
    image                   = models.ForeignKey(PlantImage, null=True, blank=True, on_delete=models.SET_NULL)
    image_uploaded          = models.BooleanField(default=False)
    predicted_height_cm     = models.FloatField(null=True, blank=True)
    predicted_yield_per_acre= models.FloatField(null=True, blank=True)
    health_score            = models.FloatField(null=True, blank=True)
    growth_stage            = models.CharField(max_length=50, blank=True)
    recommendation          = models.TextField(blank=True)
    model_accuracy          = models.FloatField(null=True, blank=True)
    sensor_data             = models.ForeignKey(SensorData, null=True, blank=True, on_delete=models.SET_NULL)

    class Meta:
        ordering = ['date', 'slot']
        unique_together = [['session','date','slot']]


class GrowthPrediction(models.Model):
    sensor_data              = models.ForeignKey(SensorData, on_delete=models.CASCADE, related_name='predictions')
    timestamp                = models.DateTimeField(auto_now_add=True)
    predicted_yield_per_acre = models.FloatField(default=0)
    growth_stage             = models.CharField(max_length=50, default='Vegetative')
    health_score             = models.FloatField(default=0.0)
    recommendation           = models.TextField(blank=True)
    model_accuracy           = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['-timestamp']
