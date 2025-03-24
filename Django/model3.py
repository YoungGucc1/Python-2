import uuid
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.db.models import Q, UniqueConstraint, CheckConstraint, F

# Abstract base model for common fields
class BaseModel(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)
    is_deleted = models.BooleanField(default=False)  # Soft delete flag

    class Meta:
        abstract = True

    def delete(self, *args, **kwargs):
        """Override delete method to implement soft delete"""
        self.is_deleted = True
        self.save()

    def hard_delete(self, *args, **kwargs):
        """Method to perform an actual delete from the database"""
        super().delete(*args, **kwargs)


# Model Managers to handle soft delete
class ActiveManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(is_deleted=False)


# User Profile extension
class UserProfile(BaseModel):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    phone = models.CharField(max_length=50, blank=True)
    address = models.TextField(blank=True)
    profile_image = models.ImageField(upload_to='profile_images/', null=True, blank=True)
    
    objects = models.Manager()  # Default manager
    active_objects = ActiveManager()  # Custom manager for non-deleted objects
    
    def __str__(self):
        return f"Profile for {self.user.username}"


# Images Model
class Image(BaseModel):
    file_path = models.ImageField(upload_to='images/')
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    resolution = models.CharField(max_length=50, blank=True)
    size = models.PositiveIntegerField(help_text="Size in KB")
    format = models.CharField(max_length=10)
    alt_text = models.CharField(max_length=255, blank=True)  # For accessibility

    objects = models.Manager()
    active_objects = ActiveManager()

    def __str__(self):
        return f"{self.file_path} ({self.format})"


# Tag Model for products
class Tag(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True)
    slug = models.SlugField(max_length=100, unique=True)
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    def __str__(self):
        return self.name


# Brand Model
class Brand(BaseModel):
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    logo = models.ForeignKey(Image, null=True, blank=True, on_delete=models.SET_NULL, related_name='brand_logos')
    website = models.URLField(blank=True)
    
    # SEO fields
    meta_title = models.CharField(max_length=255, blank=True)
    meta_description = models.TextField(blank=True)
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    class Meta:
        indexes = [models.Index(fields=['slug'])]
    
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
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    class Meta:
        verbose_name_plural = "Categories"
        indexes = [models.Index(fields=['slug'])]

    def __str__(self):
        return self.name


# Product Attributes
class ProductAttribute(BaseModel):
    name = models.CharField(max_length=100)  # e.g., "Size", "Color"
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    def __str__(self):
        return self.name


class ProductAttributeValue(BaseModel):
    attribute = models.ForeignKey(ProductAttribute, on_delete=models.CASCADE, related_name='values')
    value = models.CharField(max_length=100)  # e.g., "Red", "XL"
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    class Meta:
        unique_together = ('attribute', 'value')
    
    def __str__(self):
        return f"{self.attribute.name}: {self.value}"


# Product Model
class Product(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True)
    slug = models.SlugField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    short_description = models.CharField(max_length=255, blank=True)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, db_index=True, related_name='products')
    brand = models.ForeignKey(Brand, on_delete=models.CASCADE, null=True, blank=True, related_name='products')
    images = models.ManyToManyField(Image, related_name='products')
    tags = models.ManyToManyField(Tag, blank=True, related_name='products')
    is_featured = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    sku = models.CharField(max_length=100, unique=True)
    weight = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    has_variants = models.BooleanField(default=False)
    
    # SEO fields
    meta_title = models.CharField(max_length=255, blank=True)
    meta_description = models.TextField(blank=True)
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    class Meta:
        indexes = [
            models.Index(fields=['slug']),
            models.Index(fields=['is_active', 'is_featured']),
            models.Index(fields=['brand']),
        ]
    
    def __str__(self):
        return self.name


# Product Variants
class ProductVariant(BaseModel):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='variants')
    sku = models.CharField(max_length=100, unique=True)
    is_default = models.BooleanField(default=False)
    attribute_values = models.ManyToManyField(ProductAttributeValue, related_name='product_variants')
    weight = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    class Meta:
        indexes = [models.Index(fields=['sku'])]
    
    def __str__(self):
        return f"{self.product.name} - {self.sku}"


# Price Model (supports price history)
class Price(BaseModel):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='prices')
    variant = models.ForeignKey(ProductVariant, on_delete=models.CASCADE, null=True, blank=True, related_name='prices')
    type = models.CharField(max_length=20, choices=[('normal', 'Normal'), ('sale', 'Sale'), ('wholesale', 'Wholesale')])
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=10, default='USD')
    start_date = models.DateTimeField()
    end_date = models.DateTimeField(null=True, blank=True)  # NULL for permanent prices
    is_current = models.BooleanField(default=True)

    objects = models.Manager()
    active_objects = ActiveManager()

    class Meta:
        indexes = [models.Index(fields=['product', 'is_current'])]
        constraints = [
            # Ensure only one current price per product/variant/type combination
            UniqueConstraint(
                fields=['product', 'variant', 'type', 'currency'],
                condition=Q(is_current=True, is_deleted=False),
                name='unique_current_price'
            ),
            # Either product or variant must be set, not both
            CheckConstraint(
                check=~(Q(variant__isnull=True) & Q(product__isnull=True)),
                name='price_product_or_variant'
            )
        ]

    def __str__(self):
        target = self.variant.sku if self.variant else self.product.name
        return f"{target} - {self.amount} {self.currency}"


# Warehouse Model
class Warehouse(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True)
    description = models.TextField(blank=True)
    type = models.CharField(max_length=50, choices=[('physical', 'Physical'), ('virtual', 'Virtual')])
    address = models.TextField()
    contact_info = models.CharField(max_length=255)
    status = models.CharField(max_length=20, choices=[('active', 'Active'), ('inactive', 'Inactive')])

    objects = models.Manager()
    active_objects = ActiveManager()

    def __str__(self):
        return self.name


# Stock Model
class Stock(BaseModel):
    warehouse = models.ForeignKey(Warehouse, on_delete=models.CASCADE, related_name='stock')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='stock', null=True, blank=True)
    variant = models.ForeignKey(ProductVariant, on_delete=models.CASCADE, related_name='stock', null=True, blank=True)
    quantity = models.PositiveIntegerField()
    min_quantity = models.PositiveIntegerField(default=0)
    last_counted_at = models.DateTimeField(null=True, blank=True)

    objects = models.Manager()
    active_objects = ActiveManager()

    class Meta:
        constraints = [
            # Either product or variant must be set, not both null
            CheckConstraint(
                check=Q(variant__isnull=False) | Q(product__isnull=False),
                name='stock_product_or_variant'
            ),
            # Ensure unique stock entries per warehouse and product/variant
            UniqueConstraint(
                fields=['warehouse', 'product'],
                condition=Q(variant__isnull=True, is_deleted=False),
                name='unique_warehouse_product'
            ),
            UniqueConstraint(
                fields=['warehouse', 'variant'],
                condition=Q(product__isnull=True, is_deleted=False),
                name='unique_warehouse_variant'
            )
        ]
        indexes = [
            models.Index(fields=['warehouse', 'product']),
            models.Index(fields=['warehouse', 'variant']),
        ]

    def __str__(self):
        product_name = self.variant.product.name if self.variant else self.product.name
        return f"{product_name} - {self.quantity} in {self.warehouse.name}"


# Contragents (Customers, Suppliers, Employees)
class Contragent(BaseModel):
    user = models.OneToOneField(User, null=True, blank=True, on_delete=models.SET_NULL, related_name='contragent')
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True)
    phone = models.CharField(max_length=50, blank=True)
    email = models.EmailField(unique=True)
    type = models.CharField(max_length=50, choices=[('customer', 'Customer'), ('employee', 'Employee'), ('admin', 'Admin'), ('supplier', 'Supplier')])
    address = models.TextField(blank=True)
    tax_id = models.CharField(max_length=50, blank=True, null=True, unique=True)
    status = models.CharField(max_length=20, choices=[('active', 'Active'), ('inactive', 'Inactive')])

    objects = models.Manager()
    active_objects = ActiveManager()

    def __str__(self):
        return self.name


# Shipping and Tax Models
class ShippingMethod(BaseModel):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    is_active = models.BooleanField(default=True)
    delivery_time_min = models.PositiveIntegerField(help_text="Minimum delivery time in days", default=1)
    delivery_time_max = models.PositiveIntegerField(help_text="Maximum delivery time in days", default=3)
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    def __str__(self):
        return self.name


class TaxRate(BaseModel):
    name = models.CharField(max_length=100)
    rate = models.DecimalField(max_digits=5, decimal_places=2)  # Percentage
    country = models.CharField(max_length=100)
    state = models.CharField(max_length=100, blank=True)
    is_active = models.BooleanField(default=True)
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    class Meta:
        unique_together = ('country', 'state')
    
    def __str__(self):
        return f"{self.name} ({self.rate}%)"


# Shopping Cart
class Cart(BaseModel):
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.CASCADE, related_name='carts')
    session_id = models.CharField(max_length=255, null=True, blank=True)  # For anonymous users
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    class Meta:
        constraints = [
            CheckConstraint(
                check=Q(user__isnull=False) | Q(session_id__isnull=False),
                name='cart_user_or_session'
            )
        ]
    
    def __str__(self):
        return f"Cart {self.id}"


# Cart Items
class CartItem(BaseModel):
    cart = models.ForeignKey(Cart, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, null=True, blank=True)
    variant = models.ForeignKey(ProductVariant, on_delete=models.CASCADE, null=True, blank=True)
    quantity = models.PositiveIntegerField(default=1)
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    class Meta:
        constraints = [
            # Either product or variant must be set, not both null
            CheckConstraint(
                check=Q(variant__isnull=False) | Q(product__isnull=False),
                name='cart_item_product_or_variant'
            ),
            # Ensure unique items in cart
            UniqueConstraint(
                fields=['cart', 'product'],
                condition=Q(variant__isnull=True, is_deleted=False),
                name='unique_cart_product'
            ),
            UniqueConstraint(
                fields=['cart', 'variant'],
                condition=Q(product__isnull=True, is_deleted=False),
                name='unique_cart_variant'
            )
        ]
    
    def __str__(self):
        product_name = self.variant.product.name if self.variant else self.product.name
        return f"{product_name} in cart {self.cart.id}"


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
    shipping_method = models.ForeignKey(ShippingMethod, on_delete=models.SET_NULL, null=True, related_name='orders')
    payment_method = models.CharField(max_length=50, blank=True)
    subtotal = models.DecimalField(max_digits=10, decimal_places=2)
    tax = models.DecimalField(max_digits=10, decimal_places=2)
    tax_rate = models.ForeignKey(TaxRate, on_delete=models.SET_NULL, null=True, related_name='orders')
    shipping_cost = models.DecimalField(max_digits=10, decimal_places=2)
    discount = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    total = models.DecimalField(max_digits=10, decimal_places=2)
    notes = models.TextField(blank=True)
    tracking_number = models.CharField(max_length=100, blank=True)
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
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
    product = models.ForeignKey(Product, on_delete=models.CASCADE, null=True, blank=True)
    variant = models.ForeignKey(ProductVariant, on_delete=models.CASCADE, null=True, blank=True)
    quantity = models.PositiveIntegerField()
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)
    tax_rate = models.DecimalField(max_digits=5, decimal_places=2, help_text="Tax rate in percentage")
    tax_amount = models.DecimalField(max_digits=10, decimal_places=2)
    discount_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    total_price = models.DecimalField(max_digits=10, decimal_places=2)
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    class Meta:
        constraints = [
            # Either product or variant must be set, not both null
            CheckConstraint(
                check=Q(variant__isnull=False) | Q(product__isnull=False),
                name='order_item_product_or_variant'
            ),
        ]
    
    def __str__(self):
        product_name = self.variant.product.name if self.variant else self.product.name
        return f"{self.quantity} x {product_name} in Order {self.order.id}"


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

    objects = models.Manager()
    active_objects = ActiveManager()

    class Meta:
        verbose_name_plural = "Order status histories"
        ordering = ['-changed_at']

    def __str__(self):
        return f"{self.order.id} - {self.status} at {self.changed_at}"


# Return/Refund Models
class Return(BaseModel):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='returns')
    return_date = models.DateTimeField(auto_now_add=True)
    reason = models.TextField()
    status = models.CharField(max_length=50, choices=[
        ('pending', 'Pending'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
        ('completed', 'Completed')
    ])
    notes = models.TextField(blank=True)
    processed_by = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL, related_name='processed_returns')
    processed_date = models.DateTimeField(null=True, blank=True)
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    class Meta:
        indexes = [models.Index(fields=['order', 'status'])]
    
    def __str__(self):
        return f"Return {self.id} for Order {self.order.id} - {self.status}"


class ReturnItem(BaseModel):
    return_request = models.ForeignKey(Return, on_delete=models.CASCADE, related_name='items')
    order_item = models.ForeignKey(OrderItem, on_delete=models.CASCADE, related_name='return_items')
    quantity = models.PositiveIntegerField()
    reason = models.TextField(blank=True)
    condition = models.CharField(max_length=50, choices=[
        ('new', 'New/Unused'),
        ('opened', 'Opened'),
        ('damaged', 'Damaged'),
        ('defective', 'Defective')
    ])
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    class Meta:
        constraints = [
            # Ensure return quantity doesn't exceed original order quantity
            CheckConstraint(
                check=Q(quantity__lte=F('order_item__quantity')),
                name='return_qty_lte_order_qty'
            )
        ]
    
    def __str__(self):
        return f"Return of {self.quantity} from order item {self.order_item.id}"


class Refund(BaseModel):
    return_request = models.OneToOneField(Return, on_delete=models.CASCADE, related_name='refund')
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    refund_method = models.CharField(max_length=50, choices=[
        ('credit_card', 'Credit Card'),
        ('store_credit', 'Store Credit'),
        ('bank_transfer', 'Bank Transfer'),
        ('exchange', 'Exchange')
    ])
    status = models.CharField(max_length=50, choices=[
        ('pending', 'Pending'),
        ('processed', 'Processed'),
        ('failed', 'Failed')
    ])
    transaction_id = models.CharField(max_length=255, blank=True)
    notes = models.TextField(blank=True)
    processed_by = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    processed_date = models.DateTimeField(null=True, blank=True)
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    def __str__(self):
        return f"Refund {self.id} for Return {self.return_request.id} - {self.amount}"


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
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    def __str__(self):
        return f"Payment {self.transaction_id} for Order {self.order.id}"


# Review Model
class Review(BaseModel):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='reviews')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reviews')
    rating = models.PositiveSmallIntegerField(validators=[MinValueValidator(1), MaxValueValidator(5)])
    title = models.CharField(max_length=255, blank=True)
    comment = models.TextField(blank=True)
    is_approved = models.BooleanField(default=False)
    order_item = models.ForeignKey(OrderItem, on_delete=models.SET_NULL, null=True, blank=True, related_name='reviews')
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    class Meta:
        unique_together = ('product', 'user')
        indexes = [models.Index(fields=['product', 'is_approved'])]
    
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
    applicable_products = models.ManyToManyField(Product, blank=True, related_name='applicable_coupons')
    applicable_categories = models.ManyToManyField(Category, blank=True, related_name='applicable_coupons')
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    def __str__(self):
        return f"{self.code} - {self.discount_value}{'%' if self.discount_type == 'percentage' else ''}"


# Wishlist Model
class Wishlist(BaseModel):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='wishlists')
    name = models.CharField(max_length=255, default="Default Wishlist")
    is_public = models.BooleanField(default=False)
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    def __str__(self):
        return f"{self.name} for {self.user.username}"


class WishlistItem(BaseModel):
    wishlist = models.ForeignKey(Wishlist, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, null=True, blank=True)
    variant = models.ForeignKey(ProductVariant, on_delete=models.CASCADE, null=True, blank=True)
    added_at = models.DateTimeField(auto_now_add=True)
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    class Meta:
        constraints = [
            # Either product or variant must be set, not both null
            CheckConstraint(
                check=Q(variant__isnull=False) | Q(product__isnull=False),
                name='wishlist_item_product_or_variant'
            ),
            # Ensure unique items in wishlist
            UniqueConstraint(
                fields=['wishlist', 'product'],
                condition=Q(variant__isnull=True, is_deleted=False),
                name='unique_wishlist_product'
            ),
            UniqueConstraint(
                fields=['wishlist', 'variant'],
                condition=Q(product__isnull=True, is_deleted=False),
                name='unique_wishlist_variant'
            )
        ]
    
    def __str__(self):
        product_name = self.variant.product.name if self.variant else self.product.name
        return f"{product_name} in {self.wishlist.name}"


# Notification Model
class Notification(BaseModel):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    title = models.CharField(max_length=255)
    message = models.TextField()
    type = models.CharField(max_length=50, choices=[
        ('order', 'Order Update'),
        ('product', 'Product Update'),
        ('price', 'Price Change'),
        ('stock', 'Stock Update'),
        ('account', 'Account Update'),
        ('general', 'General')
    ])
    is_read = models.BooleanField(default=False)
    read_at = models.DateTimeField(null=True, blank=True)
    data = models.JSONField(null=True, blank=True)  # Additional context data
    
    objects = models.Manager()
    active_objects = ActiveManager()
    
    class Meta:
        indexes = [
            models.Index(fields=['user', 'is_read']),
            models.Index(fields=['user', 'type']),
        ]
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.title} for {self.user.username}"


# Notification Templates
class NotificationTemplate(BaseModel):
    name = models.CharField(max_length=100, unique=True)
    type = models.CharField(max_length=50, choices=[
        ('email', 'Email'),
        ('sms', 'SMS'),
        ('push', 'Push Notification'),
        ('in_app', 'In-App Notification')
    ])
    subject = models.CharField(max_length=255, blank=True)
    content = models.TextField()
    variables = models.JSONField(help_text="Variables that can be used in the template")
    is_active = models.BooleanField(default=True)
    
    objects = models.Manager()
    active_objects = ActiveManager()