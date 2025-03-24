import uuid
from django.db import models
from django.contrib.auth.models import User
from PIL import Image as PILImage
import os

# Abstract base model for common fields
class BaseModel(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)
    is_deleted = models.BooleanField(default=False)  # Soft delete flag

    class Meta:
        abstract = True

# User Profile extension
class UserProfile(BaseModel):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    phone = models.CharField(max_length=50, blank=True)
    address = models.TextField(blank=True)
    profile_image = models.ImageField(upload_to='profile_images/', null=True, blank=True)
    
    def __str__(self):
        return f"Profile for {self.user.username}"

# Images Model
class Image(BaseModel):
    file_path = models.ImageField(upload_to='images/')
    description = models.TextField(blank=True)
    resolution = models.CharField(max_length=50, blank=True)
    size = models.PositiveIntegerField(help_text="Size in KB", blank=True, null=True)
    format = models.CharField(max_length=10, blank=True)
    alt_text = models.CharField(max_length=255, blank=True)  # For accessibility

    def save(self, *args, **kwargs):
        # Call the parent save method to ensure the file is saved first
        super().save(*args, **kwargs)

        # Open the image using Pillow
        img = PILImage.open(self.file_path.path)

        # Automatically set resolution (width x height)
        self.resolution = f"{img.width}x{img.height}"

        # Automatically set format (e.g., JPEG, PNG)
        self.format = img.format

        # Automatically set size in KB
        file_size = os.path.getsize(self.file_path.path)  # Size in bytes
        self.size = file_size // 1024  # Convert to KB

        # Save again to update the fields
        super().save(update_fields=['resolution', 'size', 'format'])

    def __str__(self):
        return f"{self.file_path} ({self.format})"

# Brand Model (incorporating Tag functionality)
class TagBrand(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True)
    slug = models.SlugField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    logo = models.ForeignKey(Image, null=True, blank=True, on_delete=models.SET_NULL, related_name='brand_logos')
    website = models.URLField(blank=True)
    
    # SEO fields
    meta_title = models.CharField(max_length=255, blank=True)
    meta_description = models.TextField(blank=True)
    
    def __str__(self):
        return self.name

# Categories Model with hierarchy support
class Category(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True)
    slug = models.SlugField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, related_name='children')
    image = models.ForeignKey(Image, null=True, blank=True, on_delete=models.SET_NULL)
    
    # SEO fields
    meta_title = models.CharField(max_length=255, blank=True)
    meta_description = models.TextField(blank=True)
    
    class Meta:
        verbose_name_plural = "Categories"
        indexes = [models.Index(fields=['slug'])]

    def __str__(self):
        return self.name

# Product Model
class Product(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True)
    slug = models.SlugField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    short_description = models.CharField(max_length=255, blank=True)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, db_index=True, related_name='products')
    brand = models.ForeignKey(TagBrand, on_delete=models.SET_NULL, null=True, blank=True, related_name='products')
    images = models.ManyToManyField(Image, related_name='products')
    is_featured = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    sku = models.CharField(max_length=100, unique=True)
    weight = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    sale_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    
    # SEO fields
    meta_title = models.CharField(max_length=255, blank=True)
    meta_description = models.TextField(blank=True)
    
    # Tags implementation (using M2M relationship)
    tags = models.ManyToManyField(TagBrand, related_name='tagged_products', blank=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['slug']),
            models.Index(fields=['is_active', 'is_featured']),
        ]
    
    def __str__(self):
        return self.name

# Warehouse Model
class Warehouse(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True)
    description = models.TextField(blank=True)
    type = models.CharField(max_length=50, choices=[('physical', 'Physical'), ('virtual', 'Virtual')])
    address = models.TextField()
    contact_info = models.CharField(max_length=255)
    status = models.CharField(max_length=20, choices=[('active', 'Active'), ('inactive', 'Inactive')])

    def __str__(self):
        return self.name

# Stock Model
class Stock(BaseModel):
    warehouse = models.ForeignKey(Warehouse, on_delete=models.CASCADE, related_name='stock')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='stock')
    quantity = models.PositiveIntegerField()
    min_quantity = models.PositiveIntegerField(default=0)
    last_counted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ('warehouse', 'product')
        indexes = [models.Index(fields=['warehouse', 'product'])]

    def __str__(self):
        return f"{self.product.name} - {self.quantity} in {self.warehouse.name}"

# Contragents (simplified for staff and website management)
class Contragent(BaseModel):
    user = models.OneToOneField(User, null=True, blank=True, on_delete=models.SET_NULL, related_name='contragent')
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True)
    phone = models.CharField(max_length=50, blank=True)
    email = models.EmailField(unique=True)
    type = models.CharField(max_length=50, choices=[('customer', 'Customer'), ('employee', 'Employee'), ('admin', 'Admin')])
    address = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=[('active', 'Active'), ('inactive', 'Inactive')])

    def __str__(self):
        return self.name