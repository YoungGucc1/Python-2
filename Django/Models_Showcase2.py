import uuid
from django.db import models
from django.contrib.auth.models import User
from django.utils.text import slugify
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
    phone = models.CharField(max_length=50, blank=True, null=True)
    address = models.TextField(blank=True, null=True)
    profile_image = models.ImageField(upload_to='profile_images/', null=True, blank=True)
    
    def __str__(self):
        return f"Profile for {self.user.username}"

# Images Model
class Image(BaseModel):
    file_path = models.ImageField(upload_to='images/')
    description = models.TextField(blank=True, null=True)
    resolution = models.CharField(max_length=50, blank=True, null=True)
    size = models.PositiveIntegerField(help_text="Size in KB", blank=True, null=True)
    format = models.CharField(max_length=10, blank=True, null=True)
    alt_text = models.CharField(max_length=255, blank=True, null=True)  # For accessibility

    def save(self, *args, **kwargs):
        # Call parent save to ensure file is uploaded
        super().save(*args, **kwargs)

        # Open and optimize image
        img = PILImage.open(self.file_path.path)
        max_size = (1920, 1080)  # Max dimensions for optimization
        if img.width > max_size[0] or img.height > max_size[1]:
            img.thumbnail(max_size, PILImage.Resampling.LANCZOS)
            img.save(self.file_path.path, quality=85)  # Reduce quality for smaller size

        # Update metadata
        self.resolution = f"{img.width}x{img.height}"
        self.format = img.format
        self.size = os.path.getsize(self.file_path.path) // 1024  # Size in KB

        # Save updated fields
        super().save(update_fields=['resolution', 'size', 'format'])

    def __str__(self):
        return f"{self.file_path} ({self.format})"

# Brand Model
class Brand(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True, null=True)
    slug = models.SlugField(max_length=255, unique=True, blank=True)
    description = models.TextField(blank=True, null=True)
    logo = models.ForeignKey(Image, null=True, blank=True, on_delete=models.SET_NULL, related_name='brand_logos')
    website = models.URLField(blank=True, null=True)
    
    # SEO fields
    meta_title = models.CharField(max_length=255, blank=True, null=True)
    meta_description = models.TextField(blank=True, null=True)
    
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

# Categories Model with hierarchy support
class Category(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True, null=True)
    slug = models.SlugField(max_length=255, unique=True, blank=True)
    description = models.TextField(blank=True, null=True)
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, related_name='children')
    image = models.ForeignKey(Image, null=True, blank=True, on_delete=models.SET_NULL)
    
    # SEO fields
    meta_title = models.CharField(max_length=255, blank=True, null=True)
    meta_description = models.TextField(blank=True, null=True)
    
    class Meta:
        verbose_name_plural = "Categories"
        indexes = [models.Index(fields=['slug'])]

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

# Product Model
class Product(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True, null=True)
    slug = models.SlugField(max_length=255, unique=True, blank=True)
    description = models.TextField(blank=True, null=True)
    short_description = models.CharField(max_length=255, blank=True, null=True)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, db_index=True, related_name='products')
    brand = models.ForeignKey(Brand, on_delete=models.SET_NULL, null=True, blank=True, related_name='products')
    images = models.ManyToManyField(Image, related_name='products')
    is_featured = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    sku = models.CharField(max_length=100, unique=True)
    weight = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    sale_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    
    # SEO fields
    meta_title = models.CharField(max_length=255, blank=True, null=True)
    meta_description = models.TextField(blank=True, null=True)
    
    # Tags implementation
    tags = models.ManyToManyField(Brand, related_name='tagged_products', blank=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['slug']),
            models.Index(fields=['is_active', 'is_featured']),
        ]
    
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

# Warehouse Model
class Warehouse(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True, null=True)
    slug = models.SlugField(max_length=255, unique=True, blank=True)
    description = models.TextField(blank=True, null=True)
    type = models.CharField(max_length=50, choices=[('physical', 'Physical'), ('virtual', 'Virtual')])
    address = models.TextField()
    contact_info = models.CharField(max_length=255)
    status = models.CharField(max_length=20, choices=[('active', 'Active'), ('inactive', 'Inactive')])

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

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

# Contragent Model
class Contragent(BaseModel):
    user = models.OneToOneField(User, null=True, blank=True, on_delete=models.SET_NULL, related_name='contragent')
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True, null=True)
    slug = models.SlugField(max_length=255, unique=True, blank=True)
    phone = models.CharField(max_length=50, blank=True, null=True)
    email = models.EmailField(unique=True)
    type = models.CharField(max_length=50, choices=[('customer', 'Customer'), ('employee', 'Employee'), ('admin', 'Admin')])
    address = models.TextField(blank=True, null=True)
    status = models.CharField(max_length=20, choices=[('active', 'Active'), ('inactive', 'Inactive')])

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name