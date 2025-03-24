import uuid
from django.db import models
from django.contrib.auth.models import User
from PIL import Image as PILImage

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

# Categories Model with hierarchy support
class Category(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, unique=True, blank=True)
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
    name2 = models.CharField(max_length=255, unique=True, blank=True)
    slug = models.SlugField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    short_description = models.CharField(max_length=255, blank=True)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, db_index=True, related_name='products')
    images = models.ManyToManyField(Image, related_name='products')
    is_featured = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    sku = models.CharField(max_length=100, unique=True)
    weight = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    
    # SEO fields
    meta_title = models.CharField(max_length=255, blank=True)
    meta_description = models.TextField(blank=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['slug']),
            models.Index(fields=['is_active', 'is_featured']),
        ]
    
    def __str__(self):
        return self.name

# Price Model (supports price history)
class Price(BaseModel):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='prices')
    type = models.CharField(max_length=20, choices=[('normal', 'Normal'), ('sale', 'Sale'), ('wholesale', 'Wholesale')])
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=10, default='USD')
    start_date = models.DateTimeField()
    end_date = models.DateTimeField(null=True, blank=True)  # NULL for permanent prices
    is_current = models.BooleanField(default=True)

    class Meta:
        indexes = [models.Index(fields=['product', 'is_current'])]

    def __str__(self):
        return f"{self.product.name} - {self.amount} {self.currency}"

# Warehouse Model
class Warehouse(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, unique=True, blank=True)
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

# Contragents (Customers, Suppliers, Employees)
class Contragent(BaseModel):
    user = models.OneToOneField(User, null=True, blank=True, on_delete=models.SET_NULL, related_name='contragent')
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, unique=True, blank=True)
    phone = models.CharField(max_length=50, blank=True)
    email = models.EmailField(unique=True)
    type = models.CharField(max_length=50, choices=[('customer', 'Customer'), ('employee', 'Employee'), ('admin', 'Admin'), ('supplier', 'Supplier')])
    address = models.TextField(blank=True)
    tax_id = models.CharField(max_length=50, blank=True, null=True, unique=True)
    status = models.CharField(max_length=20, choices=[('active', 'Active'), ('inactive', 'Inactive')])

    def __str__(self):
        return self.name

# Shopping Cart
class Cart(BaseModel):
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.CASCADE, related_name='carts')
    session_id = models.CharField(max_length=255, null=True, blank=True)  # For anonymous users
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        constraints = [
            models.CheckConstraint(
                check=models.Q(user__isnull=False) | models.Q(session_id__isnull=False),
                name='cart_user_or_session'
            )
        ]
    
    def __str__(self):
        return f"Cart {self.id}"

# Cart Items
class CartItem(BaseModel):
    cart = models.ForeignKey(Cart, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    
    class Meta:
        unique_together = ('cart', 'product')
    
    def __str__(self):
        return f"{self.product.name} in cart {self.cart.id}"

# Orders Model
class Order(BaseModel):
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL, related_name='orders')
    contragent = models.ForeignKey(Contragent, on_delete=models.CASCADE, related_name='orders')
    order_date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=50, choices=[
        ('pending', 'Pending'), 
        ('processing', 'Processing'),
        ('shipped', 'Shipped'), 
        ('delivered', 'Delivered'), 
        ('cancelled', 'Cancelled'),
        ('refunded', 'Refunded')
    ])
    shipping_address = models.TextField()
    billing_address = models.TextField(blank=True)
    shipping_method = models.CharField(max_length=50, blank=True)
    payment_method = models.CharField(max_length=50, blank=True)
    subtotal = models.DecimalField(max_digits=10, decimal_places=2)
    tax = models.DecimalField(max_digits=10, decimal_places=2)
    shipping_cost = models.DecimalField(max_digits=10, decimal_places=2)
    discount = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    total = models.DecimalField(max_digits=10, decimal_places=2)
    notes = models.TextField(blank=True)
    tracking_number = models.CharField(max_length=100, blank=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['user']),
            models.Index(fields=['contragent']),
            models.Index(fields=['status']),
            models.Index(fields=['order_date']),
        ]

    def __str__(self):
        return f"Order {self.id} - {self.status}"

# Order Items
class OrderItem(BaseModel):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField()
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)
    total_price = models.DecimalField(max_digits=10, decimal_places=2)
    
    def __str__(self):
        return f"{self.quantity} x {self.product.name} in Order {self.order.id}"

# Order Status History (Tracking Status Changes)
class OrderStatusHistory(BaseModel):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='status_history')
    status = models.CharField(max_length=50, choices=[
        ('pending', 'Pending'), 
        ('processing', 'Processing'),
        ('shipped', 'Shipped'), 
        ('delivered', 'Delivered'), 
        ('cancelled', 'Cancelled'),
        ('refunded', 'Refunded')
    ])
    changed_at = models.DateTimeField(auto_now_add=True)
    changed_by = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    notes = models.TextField(blank=True)

    class Meta:
        verbose_name_plural = "Order status histories"
        ordering = ['-changed_at']

    def __str__(self):
        return f"{self.order.id} - {self.status} at {self.changed_at}"

# Payment Model
class Payment(BaseModel):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='payments')
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    payment_method = models.CharField(max_length=50)
    transaction_id = models.CharField(max_length=255, blank=True)
    status = models.CharField(max_length=50, choices=[
        ('pending', 'Pending'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('refunded', 'Refunded')
    ])
    payment_date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Payment {self.transaction_id} for Order {self.order.id}"

# Review Model
class Review(BaseModel):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='reviews')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reviews')
    rating = models.PositiveSmallIntegerField()
    title = models.CharField(max_length=255, blank=True)
    comment = models.TextField(blank=True)
    is_approved = models.BooleanField(default=False)
    
    class Meta:
        unique_together = ('product', 'user')
    
    def __str__(self):
        return f"Review by {self.user.username} for {self.product.name}"

# Coupon/Discount Model
class Coupon(BaseModel):
    code = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True)
    discount_type = models.CharField(max_length=20, choices=[
        ('percentage', 'Percentage'),
        ('fixed', 'Fixed Amount')
    ])
    discount_value = models.DecimalField(max_digits=10, decimal_places=2)
    min_purchase = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    valid_from = models.DateTimeField()
    valid_to = models.DateTimeField()
    is_active = models.BooleanField(default=True)
    usage_limit = models.PositiveIntegerField(null=True, blank=True)  # null means unlimited
    used_count = models.PositiveIntegerField(default=0)
    
    def __str__(self):
        return f"{self.code} - {self.discount_value}{'%' if self.discount_type == 'percentage' else ''}"

# Wishlist Model
class Wishlist(BaseModel):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='wishlists')
    products = models.ManyToManyField(Product, related_name='wishlists')
    
    def __str__(self):
        return f"Wishlist for {self.user.username}"