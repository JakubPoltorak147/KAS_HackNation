from django.db import models
from django.core.validators import FileExtensionValidator
from .utils import crop_robust_count
import os

class CleanImage(models.Model):
    title = models.CharField(max_length=100, blank=True, help_text="Np. Fiat Ducato bok")
    image = models.ImageField(
        upload_to='clean_images/',
        validators=[FileExtensionValidator(allowed_extensions=['bmp'])]
    )
    aspect_ratio = models.FloatField(default=0.0, db_index=True, editable=False)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if self.aspect_ratio == 0.0 and self.image:
            try:
                file_path = self.image.path
                cropped_img = crop_robust_count(file_path)
                if cropped_img is not None:
                    h, w = cropped_img.shape
                    new_ar = w / h
                    self.aspect_ratio = new_ar
                    super().save(update_fields=['aspect_ratio'])
            except Exception as e:
                print(f"Błąd liczenia AR: {e}")

    def __str__(self):
        return f"{self.title} (AR: {self.aspect_ratio:.2f})"

class ScanRequest(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Oczekuje'),
        ('processing', 'Przetwarzanie'),
        ('completed', 'Zakończono'),
        ('error', 'Błąd'),
    ]

    LABEL_CHOICES = [
        ('unknown', 'Nieokreślone'),
        ('clean', 'Czyste (Bez anomalii)'),
        ('dirty', 'Brudne (Przemyt/Anomalia)'),
    ]

    image = models.ImageField(
        upload_to='scans_input/',
        validators=[FileExtensionValidator(allowed_extensions=['bmp'])]
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    result_image = models.ImageField(upload_to='scans_output/', null=True, blank=True)
    anomalies_found = models.IntegerField(default=0)
    matched_clean_image = models.ForeignKey(CleanImage, null=True, blank=True, on_delete=models.SET_NULL)
    detection_data = models.JSONField(null=True, blank=True, help_text="Współrzędne ramek i wyniki pewności")
    
    difference_image = models.ImageField(
        upload_to='scans_diffs/', 
        null=True, 
        blank=True,
        verbose_name="Obraz Różnicowy (Morfologia)"
    )
    
    # Pola do labelowania
    confirmed_label = models.CharField(
        max_length=20, 
        choices=LABEL_CHOICES, 
        default='unknown',
        help_text="Czy użytkownik potwierdził, że to przemyt?"
    )
    user_comments = models.TextField(blank=True, null=True, help_text="Uwagi użytkownika do skanu")
    
    # TO JEST POLE, KTÓREGO BRAKOWAŁO:
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Skan #{self.id} [{self.confirmed_label}]"