from django.contrib import admin
from django.urls import path, include  # <--- Ważne: import include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    # To poniżej mówi: "Wszystko co wchodzi na stronę główną, wyślij do aplikacji scanner"
    path('', include('scanner.urls')), 
]

# To pozwala wyświetlać zdjęcia wgrane do folderu media (tylko w trybie developerskim)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)