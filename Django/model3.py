import uuid
from django.db import models
from django.contrib.auth.models import User
from django.utils.text import slugify
from django.utils import timezone
from django.db.models import JSONField # Import JSONField
from PIL import Image as PILImage
import os
from decimal import Decimal

# --- Helper Function for Unique Slugs ---
def generate_unique_slug(instance, source_field='name', slug_field='slug'):
    """
    Generates a unique slug for the instance.
    Appends '-<number>' if the initial slug already exists.
    """
    if getattr(instance, slug_field) and not instance._state.adding:
        # Do not regenerate slug if it already exists and we are updating
        # Or add logic here if you want slugs to update when the source_field changes
        return getattr(instance, slug_field)

    base_slug = slugify(getattr(instance, source_field))
    if not base_slug: # Handle cases where source_field might be empty
        base_slug = str(instance.id)[:8] # Use part of UUID if name is empty

    slug = base_slug
    num = 1
    ModelClass = instance.__class__

    # Check for uniqueness excluding the current instance if it's already saved
    qs = ModelClass.objects.filter(**{slug_field: slug})
    if instance.pk:
        qs = qs.exclude(pk=instance.pk)

    while qs.exists():
        slug = f"{base_slug}-{num}"
        num += 1
        qs = ModelClass.objects.filter(**{slug_field: slug})
        if instance.pk:
            qs = qs.exclude(pk=instance.pk)

    return slug

# --- Abstract Base Model ---
class BaseModel(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True) # Index creation time
    modified_at = models.DateTimeField(auto_now=True)
    is_deleted = models.BooleanField(default=False, db_index=True) # Index soft delete flag

    class Meta:
        abstract = True
        ordering = ['-created_at']

# --- User Profile ---
class UserProfile(BaseModel):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    phone = models.CharField(max_length=50, blank=True, null=True)
    address = models.TextField(blank=True, null=True)
    profile_image = models.ImageField(upload_to='profile_images/', null=True, blank=True)

    def __str__(self):
        return f"Profile for {self.user.username}"

# --- Image Model ---
class Image(BaseModel):
    file_path = models.ImageField(upload_to='images/')
    description = models.TextField(blank=True, null=True)
    resolution = models.CharField(max_length=50, blank=True, null=True)
    size = models.PositiveIntegerField(help_text="Size in KB", blank=True, null=True)
    format = models.CharField(max_length=10, blank=True, null=True)
    alt_text = models.CharField(max_length=255, blank=True, null=True, help_text="Text description for accessibility and SEO")

    def save(self, *args, **kwargs):
        # Note: Image processing remains synchronous as requested.
        # Consider async processing (e.g., Celery) for better performance in production.
        is_new = self._state.adding
        super().save(*args, **kwargs) # Save first to ensure file path is set

        if (is_new or 'file_path' in kwargs.get('update_fields', [])) and self.file_path:
            try:
                img_path = self.file_path.path
                img = PILImage.open(img_path)

                # --- Image Optimization ---
                max_size = (1920, 1080)
                img_changed = False
                current_format = img.format # Store original format

                # Resize if larger than max_size
                if img.width > max_size[0] or img.height > max_size[1]:
                    img.thumbnail(max_size, PILImage.Resampling.LANCZOS)
                    img_changed = True

                # Convert to RGB if necessary (e.g., from RGBA or P) for better JPEG compatibility
                if img.mode not in ('RGB', 'L'): # L is grayscale
                    img = img.convert('RGB')
                    img_changed = True # Format potentially changes if converted

                # Save with optimization
                save_kwargs = {'quality': 85, 'optimize': True}
                # Retain format if possible, default to JPEG if format unknown or problematic
                save_format = current_format if current_format in ['JPEG', 'PNG', 'GIF'] else 'JPEG'
                if img_changed:
                    # Overwrite the original file
                    img.save(img_path, format=save_format, **save_kwargs)

                # --- Update Metadata ---
                # Re-open the potentially modified image to get final stats
                with PILImage.open(img_path) as final_img:
                    self.resolution = f"{final_img.width}x{final_img.height}"
                    self.format = final_img.format
                    self.size = os.path.getsize(img_path) // 1024 # Size in KB

                # Save updated metadata fields without triggering recursion
                Image.objects.filter(pk=self.pk).update(resolution=self.resolution, size=self.size, format=self.format)

            except FileNotFoundError:
                print(f"Warning: Could not process image {self.file_path.path}, file not found.")
            except Exception as e:
                # Consider using proper logging instead of print
                print(f"Error processing image {self.file_path.path}: {e}")

    def __str__(self):
        if self.file_path:
            base_name = os.path.basename(self.file_path.name)
            return f"{base_name} ({self.format})" if self.format else base_name
        return f"Image {self.id}"

# --- Brand Model ---
class Brand(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True, null=True, help_text="Russian name (or alternative)")
    slug = models.SlugField(max_length=255, unique=True, blank=True)
    description = models.TextField(blank=True, null=True)
    logo = models.ForeignKey(Image, null=True, blank=True, on_delete=models.SET_NULL, related_name='brand_logos')
    website = models.URLField(blank=True, null=True)
    meta_title = models.CharField(max_length=255, blank=True, null=True)
    meta_description = models.TextField(blank=True, null=True)

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = generate_unique_slug(self, 'name', 'slug')
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

# --- Category Model ---
class Category(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True, null=True, help_text="Russian name (or alternative)")
    slug = models.SlugField(max_length=255, unique=True, blank=True)
    description = models.TextField(blank=True, null=True)
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, related_name='children', db_index=True)
    image = models.ForeignKey(Image, null=True, blank=True, on_delete=models.SET_NULL, related_name='category_images')
    meta_title = models.CharField(max_length=255, blank=True, null=True)
    meta_description = models.TextField(blank=True, null=True)

    class Meta(BaseModel.Meta):
        verbose_name_plural = "Categories"
        indexes = [models.Index(fields=['slug'])]

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = generate_unique_slug(self, 'name', 'slug')
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

# --- Product Model (Grouping for Variants) ---
class Product(BaseModel):
    name = models.CharField(max_length=255, db_index=True)
    name2 = models.CharField(max_length=255, blank=True, null=True, help_text="Russian name (or alternative)")
    slug = models.SlugField(max_length=255, unique=True, blank=True)
    description = models.TextField(blank=True, null=True)
    short_description = models.CharField(max_length=255, blank=True, null=True)
    category = models.ForeignKey(Category, on_delete=models.PROTECT, db_index=True, related_name='products')
    brand = models.ForeignKey(Brand, on_delete=models.PROTECT, null=True, blank=True, related_name='products')
    # Images, price, sku, weight etc. are moved to ProductVariant
    is_featured = models.BooleanField(default=False, db_index=True)
    is_active = models.BooleanField(default=True, db_index=True, help_text="Disable this to hide the product and all its variants")
    meta_title = models.CharField(max_length=255, blank=True, null=True)
    meta_description = models.TextField(blank=True, null=True)

    class Meta(BaseModel.Meta):
        indexes = [
            models.Index(fields=['slug']),
            models.Index(fields=['is_active', 'is_featured']),
            models.Index(fields=['name']),
            models.Index(fields=['category']),
            models.Index(fields=['brand']),
        ]
        ordering = ['name'] # Often useful to order products by name

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = generate_unique_slug(self, 'name', 'slug')
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

    @property
    def default_variant(self):
        """Returns the first active variant, or None."""
        return self.variants.filter(is_active=True).first()

    @property
    def current_price(self):
        """Returns the price of the default variant, if available."""
        variant = self.default_variant
        return variant.current_price if variant else None


# --- Product Variant Model ---
class ProductVariant(BaseModel):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='variants')
    sku = models.CharField(max_length=100, unique=True, db_index=True, help_text="Unique Stock Keeping Unit for this variant")
    variant_name = models.CharField(max_length=255, blank=True, null=True, help_text="e.g., 'Red, Large' or '10kg Bag'")
    price = models.DecimalField(max_digits=10, decimal_places=2, help_text="Base price per unit")
    sale_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True, help_text="Optional discounted price")
    weight = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True, help_text="Weight for shipping calculation")
    is_active = models.BooleanField(default=True, db_index=True, help_text="Is this specific variant available for sale?")
    attributes = JSONField(blank=True, null=True, help_text='Variant attributes like {"color": "Red", "size": "L"}')
    images = models.ManyToManyField(Image, related_name='product_variants', blank=True)

    class Meta(BaseModel.Meta):
        unique_together = (('product', 'sku'),) # SKU should be unique overall, this adds extra safety per product
        indexes = [
            models.Index(fields=['sku']),
            models.Index(fields=['is_active']),
        ]
        ordering = ['product__name', 'variant_name'] # Order by product then variant

    def __str__(self):
        name_parts = [self.product.name]
        if self.variant_name:
            name_parts.append(f"({self.variant_name})")
        elif self.sku:
             name_parts.append(f"[{self.sku}]")
        return " ".join(name_parts)

    @property
    def display_name(self):
        """ A more complete name for display purposes. """
        if self.variant_name:
            return f"{self.product.name} - {self.variant_name}"
        return self.product.name # Fallback if no variant name

    @property
    def current_price(self):
        """ Returns the active price (sale price if available and lower). """
        if self.sale_price is not None and self.sale_price < self.price:
            return self.sale_price
        return self.price

# --- Warehouse Model ---
class Warehouse(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True, null=True, help_text="Russian name (or alternative)")
    slug = models.SlugField(max_length=255, unique=True, blank=True)
    description = models.TextField(blank=True, null=True)
    type = models.CharField(max_length=50, choices=[('physical', 'Physical'), ('virtual', 'Virtual')], default='physical')
    address = models.TextField(blank=True, null=True)
    contact_info = models.CharField(max_length=255, blank=True, null=True)
    status = models.CharField(max_length=20, choices=[('active', 'Active'), ('inactive', 'Inactive')], default='active', db_index=True)

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = generate_unique_slug(self, 'name', 'slug')
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

# --- Stock Model (Inventory per Variant per Warehouse) ---
class Stock(BaseModel):
    warehouse = models.ForeignKey(Warehouse, on_delete=models.CASCADE, related_name='stock_items')
    product_variant = models.ForeignKey(ProductVariant, on_delete=models.CASCADE, related_name='stock_levels') # Link to specific variant
    quantity = models.IntegerField(default=0, help_text="Current quantity on hand")
    min_quantity = models.PositiveIntegerField(default=0, help_text="Minimum desired quantity (for reordering reports)")
    last_counted_at = models.DateTimeField(null=True, blank=True)

    class Meta(BaseModel.Meta):
        unique_together = ('warehouse', 'product_variant') # One stock record per variant per warehouse
        indexes = [models.Index(fields=['warehouse', 'product_variant'])]

    def __str__(self):
        return f"{self.product_variant.display_name} ({self.quantity}) in {self.warehouse.name}"

# --- Contragent Model (Customer, Supplier, etc.) ---
class Contragent(BaseModel):
    user = models.OneToOneField(User, null=True, blank=True, on_delete=models.SET_NULL, related_name='contragent_profile')
    name = models.CharField(max_length=255, db_index=True)
    name2 = models.CharField(max_length=255, blank=True, null=True, help_text="Russian name (or alternative)")
    # Slug is not unique here, can be used for internal linking/grouping if needed
    slug = models.SlugField(max_length=255, blank=True, db_index=True) # Index non-unique slug if used for filtering
    phone = models.CharField(max_length=50, blank=True, null=True)
    # Email is not unique, allowing multiple contacts without email or sharing secondary emails
    email = models.EmailField(null=True, blank=True, db_index=True)
    type = models.CharField(max_length=50, choices=[('customer', 'Customer'), ('supplier', 'Supplier'), ('employee', 'Employee'), ('other', 'Other')], default='customer', db_index=True)
    address = models.TextField(blank=True, null=True, help_text="Default billing/shipping address (consider structured address model for complex needs)")
    status = models.CharField(max_length=20, choices=[('active', 'Active'), ('inactive', 'Inactive')], default='active', db_index=True)
    company_name = models.CharField(max_length=255, blank=True, null=True)
    vat_id = models.CharField(max_length=50, blank=True, null=True, db_index=True) # Index VAT ID if searchable

    def save(self, *args, **kwargs):
        if not self.slug and self.name:
            # Generate a non-unique slug
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.name} ({self.get_type_display()})"


# --- Sale (Order) Model ---
class Sale(BaseModel):
    SALE_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('shipped', 'Shipped'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
        ('refunded', 'Refunded'),
    ]
    PAYMENT_STATUS_CHOICES = [
        ('unpaid', 'Unpaid'),
        ('paid', 'Paid'),
        ('partial', 'Partially Paid'),
        ('refunded', 'Refunded'), # Added refunded status
    ]

    sale_number = models.CharField(max_length=50, unique=True, blank=True, db_index=True, help_text="Unique identifier for the sale")
    customer = models.ForeignKey(Contragent, on_delete=models.PROTECT, related_name='sales', limit_choices_to={'type': 'customer'}, db_index=True)
    sale_date = models.DateTimeField(default=timezone.now, db_index=True)
    status = models.CharField(max_length=20, choices=SALE_STATUS_CHOICES, default='pending', db_index=True)

    shipping_address = models.TextField(blank=True, null=True, help_text="Shipping address at the time of sale")
    billing_address = models.TextField(blank=True, null=True, help_text="Billing address at the time of sale")

    # Financials - What the customer pays
    discount_amount = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal('0.00'), help_text="Total discount applied to the sale")
    # Add shipping cost if applicable
    # shipping_cost = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0.00'))

    # Payment details
    payment_method = models.CharField(max_length=50, blank=True, null=True)
    payment_status = models.CharField(max_length=20, choices=PAYMENT_STATUS_CHOICES, default='unpaid', db_index=True)
    transaction_id = models.CharField(max_length=100, blank=True, null=True, db_index=True, help_text="Payment gateway transaction ID")

    # Deductions (Store percentages, calculate amounts via properties)
    # These represent costs incurred by the seller *from* the sale amount
    tax_percentage = models.DecimalField(max_digits=5, decimal_places=2, default=Decimal('0.00'), help_text="e.g., Sales tax percentage (3.00 for 3%)")
    acquiring_percentage = models.DecimalField(max_digits=5, decimal_places=2, default=Decimal('0.00'), help_text="e.g., Payment gateway fee percentage (0.95 for 0.95%)")
    commission_percentage = models.DecimalField(max_digits=5, decimal_places=2, default=Decimal('0.00'), help_text="e.g., Salesperson commission percentage (10.00 for 10%)")

    notes = models.TextField(blank=True, null=True, help_text="Internal notes or customer comments")

    class Meta(BaseModel.Meta):
        verbose_name = "Sale / Order"
        verbose_name_plural = "Sales / Orders"
        indexes = [
            models.Index(fields=['sale_number']),
            models.Index(fields=['customer', 'sale_date']),
            models.Index(fields=['status']),
            models.Index(fields=['payment_status']),
        ]

    def save(self, *args, **kwargs):
        # Generate unique sale number *before* first save
        if not self.sale_number:
            timestamp = timezone.now().strftime('%Y%m%d%H%M') # Use minute precision
            unique_part = uuid.uuid4().hex[:6].upper() # 6 hex chars = ~16.7M possibilities
            potential_number = f"S-{timestamp}-{unique_part}"
            # Ensure uniqueness (very low chance of collision, but check anyway)
            while Sale.objects.filter(sale_number=potential_number).exists():
                unique_part = uuid.uuid4().hex[:6].upper()
                potential_number = f"S-{timestamp}-{unique_part}"
            self.sale_number = potential_number

        # Optionally copy addresses from customer if not provided
        if not self.billing_address and self.customer and self.customer.address:
             self.billing_address = self.customer.address
        # Add logic for shipping address if customer has separate shipping addr field/model

        super().save(*args, **kwargs)

    # --- Calculated Properties ---

    @property
    def subtotal(self):
        """Calculates the subtotal from all sale items *before* sale-level discounts."""
        return sum(item.line_total for item in self.items.all() if item.line_total is not None)

    @property
    def total_amount(self):
        """Calculates the final amount paid by the customer (subtotal - discounts + shipping etc.)."""
        # Add + self.shipping_cost here if you add that field
        return self.subtotal - self.discount_amount

    @property
    def tax_deduction(self):
        """Calculates the tax amount deducted from the total amount."""
        if self.tax_percentage > 0:
            # Usually tax is calculated on the amount *after* discounts
            return (self.total_amount * (self.tax_percentage / Decimal('100'))).quantize(Decimal('0.01'))
        return Decimal('0.00')

    @property
    def acquiring_deduction(self):
        """Calculates the acquiring fee deducted from the total amount."""
        if self.acquiring_percentage > 0:
            return (self.total_amount * (self.acquiring_percentage / Decimal('100'))).quantize(Decimal('0.01'))
        return Decimal('0.00')

    @property
    def commission_deduction(self):
        """Calculates the commission deducted from the total amount."""
        # Clarify if commission is based on total_amount, subtotal, or net revenue
        # Assuming total_amount for now:
        if self.commission_percentage > 0:
            return (self.total_amount * (self.commission_percentage / Decimal('100'))).quantize(Decimal('0.01'))
        return Decimal('0.00')

    @property
    def total_deductions(self):
        """Calculates the sum of all seller-side deductions."""
        return self.tax_deduction + self.acquiring_deduction + self.commission_deduction

    @property
    def net_revenue(self):
        """Calculates the revenue remaining after all deductions."""
        return self.total_amount - self.total_deductions

    def __str__(self):
        return f"Sale {self.sale_number} ({self.customer.name})"


# --- Sale Item Model (Line Item within a Sale) ---
class SaleItem(BaseModel):
    sale = models.ForeignKey(Sale, on_delete=models.CASCADE, related_name='items', db_index=True)
    # Link to the specific variant sold
    product_variant = models.ForeignKey(ProductVariant, on_delete=models.PROTECT, related_name='sale_items')
    quantity = models.PositiveIntegerField(default=1)

    # Store details AT THE TIME OF SALE
    product_name = models.CharField(max_length=255, blank=True, help_text="Product name at time of sale (incl. variant)")
    sku_at_sale = models.CharField(max_length=100, blank=True, null=True, db_index=True, help_text="Variant SKU at time of sale")
    price_at_sale = models.DecimalField(max_digits=10, decimal_places=2, help_text="Price per unit at time of sale")

    class Meta(BaseModel.Meta):
        # Allow the same variant multiple times? Usually no, adjust quantity.
        # If variants can have options chosen at sale time (e.g. engraving), this might change.
        unique_together = ('sale', 'product_variant')
        verbose_name = "Sale Item"
        verbose_name_plural = "Sale Items"
        ordering = ['created_at'] # Order items by addition time within a sale

    def save(self, *args, **kwargs):
        # Capture product/variant details if not already set (e.g., when creating the item)
        if self.product_variant and not self.product_name:
            self.product_name = self.product_variant.display_name # Use the combined name
        if self.product_variant and not self.sku_at_sale:
            self.sku_at_sale = self.product_variant.sku
        # Price should ideally be set explicitly when the item is added based on current price/promos
        # If price_at_sale is missing, we might default it, but it's safer to require it.
        # if self.product_variant and self.price_at_sale is None:
        #     self.price_at_sale = self.product_variant.current_price

        super().save(*args, **kwargs)
        # Note: If you were storing calculated totals on Sale, you'd need to trigger
        # an update on self.sale here (potentially using signals for cleaner separation).
        # Since we are using properties on Sale, this is not strictly needed for calculation.

    @property
    def line_total(self):
        """Calculates the total for this line item (quantity * price)."""
        if self.price_at_sale is not None and self.quantity is not None:
            return self.quantity * self.price_at_sale
        return Decimal('0.00') # Return Decimal for consistency

    def __str__(self):
        p_name = self.product_name or (self.product_variant.display_name if self.product_variant else "N/A")
        return f"{self.quantity} x {p_name} in Sale {self.sale.sale_number}"

# --- END OF REVISED FILE model3.py ---