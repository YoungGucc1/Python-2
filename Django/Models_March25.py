import uuid
from django.db import models

# Abstract base model for common fields
class BaseModel(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

# Images Model
class Image(BaseModel):
    file_path = models.ImageField(upload_to='images/')
    description = models.TextField(blank=True)
    resolution = models.CharField(max_length=50, blank=True)
    size = models.PositiveIntegerField(help_text="Size in KB")
    format = models.CharField(max_length=10)

    def __str__(self):
        return f"{self.file_path} ({self.format})"

# Categories Model
class Category(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True)

    def __str__(self):
        return self.name

# Product Model
class Product(BaseModel):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, db_index=True)
    images = models.ManyToManyField(Image, related_name='products')

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

    def __str__(self):
        return f"{self.product.name} - {self.amount} {self.currency}"

# Warehouse Model
class Warehouse(BaseModel):
    name = models.CharField(max_length=255)
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

    def __str__(self):
        return f"{self.product.name} - {self.quantity} in {self.warehouse.name}"

# Contragents (Customers, Suppliers, Employees)
class Contragent(BaseModel):
    name = models.CharField(max_length=255)
    phone = models.CharField(max_length=50, blank=True)
    email = models.EmailField(unique=True)
    type = models.CharField(max_length=50, choices=[('customer', 'Customer'), ('employee', 'Employee'), ('admin', 'Admin'), ('supplier', 'Supplier')])
    address = models.TextField(blank=True)
    tax_id = models.CharField(max_length=50, blank=True, null=True, unique=True)
    status = models.CharField(max_length=20, choices=[('active', 'Active'), ('inactive', 'Inactive')])

    def __str__(self):
        return self.name

# Orders Model
class Order(BaseModel):
    contragent = models.ForeignKey(Contragent, on_delete=models.CASCADE, related_name='orders')
    order_date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=50, choices=[('pending', 'Pending'), ('shipped', 'Shipped'), ('delivered', 'Delivered'), ('cancelled', 'Cancelled')])
    shipping_address = models.TextField()
    shipping_method = models.CharField(max_length=50, blank=True)
    payment_method = models.CharField(max_length=50, blank=True)
    subtotal = models.DecimalField(max_digits=10, decimal_places=2)
    tax = models.DecimalField(max_digits=10, decimal_places=2)
    shipping_cost = models.DecimalField(max_digits=10, decimal_places=2)
    total = models.DecimalField(max_digits=10, decimal_places=2)
    notes = models.TextField(blank=True)

    def __str__(self):
        return f"Order {self.id} - {self.status}"

# Order Status History (Tracking Status Changes)
class OrderStatusHistory(BaseModel):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='status_history')
    status = models.CharField(max_length=50, choices=[('pending', 'Pending'), ('shipped', 'Shipped'), ('delivered', 'Delivered'), ('cancelled', 'Cancelled')])
    changed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.order.id} - {self.status} at {self.changed_at}"