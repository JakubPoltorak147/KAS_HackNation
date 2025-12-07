from django import forms
from .models import ScanRequest

class ScanRequestForm(forms.ModelForm):
    class Meta:
        model = ScanRequest
        fields = ['image'] # Użytkownik podaje tylko obrazek
        widgets = {
            'image': forms.FileInput(attrs={
                'class': 'file-input', 
                'accept': 'image/*',
                'id': 'fileInput'
            })
        }