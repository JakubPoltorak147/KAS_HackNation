from django.contrib import admin
from .models import CleanImage, ScanRequest

@admin.register(CleanImage)
class CleanImageAdmin(admin.ModelAdmin):
    list_display = ('title', 'aspect_ratio', 'uploaded_at')
    search_fields = ('title',)

@admin.register(ScanRequest)
class ScanRequestAdmin(admin.ModelAdmin):
    # Tutaj dodajemy 'confirmed_label' i 'user_comments' do listy kolumn
    list_display = ('id', 'status', 'anomalies_found', 'confirmed_label', 'user_comments', 'created_at')
    
    # Dodajemy filtr po prawej stronie, żebyś mógł filtrować np. tylko "Potwierdzone Przemyty"
    list_filter = ('status', 'confirmed_label')
    
    # Dodajemy wyszukiwarkę (np. po komentarzu)
    search_fields = ('user_comments',)
    
    # Opcjonalnie: ustawiamy te pola jako tylko do odczytu w detalu, żeby admin ich przypadkiem nie zmienił
    readonly_fields = ('created_at', 'result_image')