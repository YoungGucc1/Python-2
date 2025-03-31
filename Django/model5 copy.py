# --- START OF FILE models.py ---

import uuid
import os
import logging
from decimal import Decimal

from django.conf import settings # Use settings for AUTH_USER_MODEL etc.
from django.db import models, transaction
from django.db.models import F, Q, Sum, UniqueConstraint # Import Q, Sum, UniqueConstraint
# from django.contrib.auth.models import User # Use settings.AUTH_USER_MODEL instead
from django.utils.text import slugify
from django.utils import timezone
from django.db.models import JSONField
from PIL import Image as PILImage

logger = logging.getLogger(__name__) # Setup logger

# --- Helper Function for Unique Slugs ---
# (Assuming generate_unique_slug function remains the same as provided previously)
def generate_unique_slug(instance, source_field='name', slug_field='slug'):
    """
    Generates a unique slug for the instance.
    Appends '-<number>' if the initial slug already exists.
    """
    if getattr(instance, slug_field) and not instance._state.adding:
        # Do not regenerate slug if it already exists and we are updating
        return getattr(instance, slug_field)

    source_value = getattr(instance, source_field)
    if not source_value: # Handle cases where source_field might be empty
         # Generate slug from first 8 chars of UUID if name is empty and instance has id
         base_slug = str(instance.id)[:8] if instance.id else uuid.uuid4().hex[:8]
    else:
        base_slug = slugify(source_value)

    slug = base_slug
    num = 1
    ModelClass = instance.__class__

    # Use the correct manager (objects or all_objects depending on context)
    # Defaulting to 'objects' which respects soft delete if applicable
    manager = getattr(ModelClass, 'objects', ModelClass._default_manager)
    qs = manager.filter(**{slug_field: slug})

    if instance.pk:
        qs = qs.exclude(pk=instance.pk)

    while qs.exists():
        slug = f"{base_slug}-{num}"
        num += 1
        qs = manager.filter(**{slug_field: slug})
        if instance.pk:
            qs = qs.exclude(pk=instance.pk)

    return slug


# --- Custom Manager for Soft Delete ---
class SoftDeleteManager(models.Manager):
    def get_queryset(self):
        # Default manager only returns non-deleted objects
        return super().get_queryset().filter(is_deleted=False)

class AllObjectsManager(models.Manager):
    def get_queryset(self):
        # Manager to return all objects, including deleted ones
        return super().get_queryset()


# --- Abstract Base Models ---
class BaseModel(models.Model):
    """ Base model with UUID and timestamps, no soft delete """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    modified_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True
        ordering = ['-created_at']

class SoftDeleteModel(BaseModel):
    """ Base model with UUID, timestamps, and soft delete functionality """
    is_deleted = models.BooleanField(default=False, db_index=True)

    # Managers
    objects = SoftDeleteManager() # Default manager, returns only active objects
    all_objects = AllObjectsManager() # Returns all objects

    class Meta:
        abstract = True
        ordering = ['-created_at']

    def soft_delete(self):
        """Marks the instance as deleted."""
        self.is_deleted = True
        self.save(update_fields=['is_deleted', 'modified_at'])

    def restore(self):
        """Restores a soft-deleted instance."""
        self.is_deleted = False
        self.save(update_fields=['is_deleted', 'modified_at'])


# --- Structured Address Model ---
class Address(SoftDeleteModel): # Inherits soft delete
    # Consider using django-countries or similar for standardized country field
    country = models.CharField(max_length=100, blank=True, null=True)
    region = models.CharField(max_length=150, blank=True, null=True, help_text="State / Province / Region")
    city = models.CharField(max_length=100, blank=True, null=True)
    street_address = models.CharField(max_length=255, blank=True, null=True, help_text="Street name and house number")
    apartment_suite = models.CharField(max_length=100, blank=True, null=True, help_text="Apartment, suite, unit, building, floor, etc.")
    postal_code = models.CharField(max_length=20, blank=True, null=True, db_index=True)
    # Could add: latitude, longitude, delivery_instructions

    class Meta(SoftDeleteModel.Meta):
        verbose_name = "Address"
        verbose_name_plural = "Addresses"
        # unique_together removed - Addresses often aren't globally unique

    def __str__(self):
        parts = [
            self.street_address,
            self.apartment_suite,
            self.city,
            self.region,
            self.postal_code,
            self.country
        ]
        return ", ".join(filter(None, parts)) or f"Address {self.id}"

# --- User Profile ---
class UserProfile(SoftDeleteModel): # Inherits soft delete
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, # Use configured user model
        on_delete=models.CASCADE,
        related_name='profile'
    )
    phone = models.CharField(max_length=50, blank=True, null=True)
    # Use ForeignKey to the structured Address model
    address = models.ForeignKey(
        Address,
        on_delete=models.SET_NULL, # Keep profile if address deleted (can be re-assigned)
        null=True, blank=True,
        related_name='user_profiles'
    )
    profile_image = models.ImageField(upload_to='profile_images/', null=True, blank=True)

    def __str__(self):
        return f"Profile for {self.user.username}"

# --- Image Model ---
class Image(SoftDeleteModel): # Inherits soft delete (optional, could be BaseModel)
    file_path = models.ImageField(upload_to='images/')
    description = models.TextField(blank=True, null=True)
    # Metadata fields, populated by the async task
    resolution = models.CharField(max_length=50, blank=True, null=True, editable=False)
    size = models.PositiveIntegerField(help_text="Size in KB", blank=True, null=True, editable=False)
    format = models.CharField(max_length=10, blank=True, null=True, editable=False)
    alt_text = models.CharField(max_length=255, blank=True, null=True, help_text="Text description for accessibility and SEO")

    def save(self, *args, **kwargs):
        # --- !!! ASYNCHRONOUS PROCESSING REQUIRED !!! ---
        # The actual image processing (resizing, optimization, metadata extraction)
        # MUST be moved to a background task (e.g., using Celery, Django Q, RQ).
        # The task should be triggered *after* this initial save.
        # The task will then update the resolution, size, and format fields.
        # The task should also handle potential file renaming if the format changes.

        super().save(*args, **kwargs)

        # Example: Trigger async task (pseudo-code)
        # if not settings.TESTING: # Avoid triggering in tests unless needed
        #    from .tasks import process_image_task # Import your task
        #    process_image_task.delay(self.pk)

    def __str__(self):
        if self.file_path:
            base_name = os.path.basename(self.file_path.name)
            return f"{base_name} ({self.format})" if self.format else base_name
        return f"Image {self.id}"

# --- Brand Model ---
class Brand(SoftDeleteModel): # Inherits soft delete
    name = models.CharField(max_length=255, unique=True)
    # Consider i18n library like django-parler for multi-language support instead of name2
    name2 = models.CharField(max_length=255, blank=True, null=True, help_text="Russian name (or alternative)")
    slug = models.SlugField(max_length=255, unique=True, blank=True, help_text="Leave blank to auto-generate")
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
class Category(SoftDeleteModel): # Inherits soft delete
    name = models.CharField(max_length=255, unique=True)
    # Consider i18n library like django-parler for multi-language support instead of name2
    name2 = models.CharField(max_length=255, blank=True, null=True, help_text="Russian name (or alternative)")
    slug = models.SlugField(max_length=255, unique=True, blank=True, help_text="Leave blank to auto-generate")
    description = models.TextField(blank=True, null=True)
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, related_name='children', db_index=True)
    image = models.ForeignKey(Image, null=True, blank=True, on_delete=models.SET_NULL, related_name='category_images')
    meta_title = models.CharField(max_length=255, blank=True, null=True)
    meta_description = models.TextField(blank=True, null=True)

    class Meta(SoftDeleteModel.Meta):
        verbose_name_plural = "Categories"
        indexes = [models.Index(fields=['slug'])]

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = generate_unique_slug(self, 'name', 'slug')
        # Prevent setting parent to self
        if self.pk and self.parent_id == self.pk:
             raise ValueError("A category cannot be its own parent.")
        # Optional: Prevent circular dependencies (more complex validation needed if allowing deep hierarchies)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

# --- Product Model (Grouping for Variants) ---
class Product(SoftDeleteModel): # Inherits soft delete
    name = models.CharField(max_length=255, db_index=True)
    # Consider i18n library like django-parler for multi-language support instead of name2
    name2 = models.CharField(max_length=255, blank=True, null=True, help_text="Russian name (or alternative)")
    slug = models.SlugField(max_length=255, unique=True, blank=True, help_text="Leave blank to auto-generate")
    description = models.TextField(blank=True, null=True)
    short_description = models.CharField(max_length=255, blank=True, null=True)
    category = models.ForeignKey(Category, on_delete=models.PROTECT, db_index=True, related_name='products')
    brand = models.ForeignKey(Brand, on_delete=models.PROTECT, null=True, blank=True, related_name='products')
    is_featured = models.BooleanField(default=False, db_index=True)
    # Renamed from is_active to avoid clash with SoftDeleteModel.is_active property if added
    is_published = models.BooleanField(default=True, db_index=True, help_text="Controls overall visibility of the product group")
    meta_title = models.CharField(max_length=255, blank=True, null=True)
    meta_description = models.TextField(blank=True, null=True)

    class Meta(SoftDeleteModel.Meta):
        indexes = [
            models.Index(fields=['slug']),
            models.Index(fields=['is_published', 'is_featured']),
        ]
        ordering = ['name']

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = generate_unique_slug(self, 'name', 'slug')
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

    @property
    def default_variant(self):
        """Returns the explicitly marked default variant, or the first active one as fallback."""
        # Use prefetch_related('variants') in views/serializers for efficiency
        default = self.variants.filter(is_active=True, is_default=True).first()
        if not default:
            # Fallback if no default is explicitly set
            default = self.variants.filter(is_active=True).order_by('created_at').first()
        return default

    @property
    def current_price(self):
        """Returns the price of the default variant, if available."""
        # Use prefetch_related('variants') in views/serializers for efficiency
        variant = self.default_variant
        return variant.current_price if variant else None


# --- Product Variant Model ---
class ProductVariant(SoftDeleteModel): # Inherits soft delete
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='variants')
    sku = models.CharField(max_length=100, unique=True, db_index=True, help_text="Unique Stock Keeping Unit for this variant")
    variant_name = models.CharField(max_length=255, blank=True, null=True, help_text="e.g., 'Red, Large' or '10kg Bag'. Distinguishes from other variants.")
    price = models.DecimalField(max_digits=12, decimal_places=2, help_text="Base price per unit")
    sale_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True, help_text="Optional discounted price")
    weight = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True, help_text="Weight (e.g., in kg) for shipping calculation")
    is_active = models.BooleanField(default=True, db_index=True, help_text="Is this specific variant available for sale?")
    is_default = models.BooleanField(default=False, help_text="Is this the default variant for the product? Only one per product.")

    # JSONField is flexible. For robust filtering on attributes, consider EAV or dedicated Attribute models.
    attributes = JSONField(blank=True, null=True, help_text='Variant attributes like {"color": "Red", "size": "L"}')
    images = models.ManyToManyField(Image, related_name='product_variants', blank=True)

    class Meta(SoftDeleteModel.Meta):
        indexes = [
            models.Index(fields=['sku']),
            models.Index(fields=['product', 'is_active']),
            models.Index(fields=['product', 'is_default']),
        ]
        # Ensure only one variant can be default per product
        constraints = [
            UniqueConstraint(fields=['product'], condition=Q(is_default=True), name='unique_default_variant_per_product')
        ]
        ordering = ['product__name', 'variant_name', 'sku']

    def save(self, *args, **kwargs):
        # If this variant is being set as default, ensure others are not
        if self.is_default:
            # Use _base_manager to bypass soft delete manager if needed, though 'objects' should be fine here
            ProductVariant.objects.filter(product=self.product, is_default=True).exclude(pk=self.pk).update(is_default=False)
        super().save(*args, **kwargs)


    def __str__(self):
        name_parts = [self.product.name]
        if self.variant_name:
            name_parts.append(f"({self.variant_name})")
        name_parts.append(f"[{self.sku}]")
        return " ".join(name_parts)

    @property
    def display_name(self):
        """ A more complete name for display purposes. """
        if self.variant_name:
            return f"{self.product.name} - {self.variant_name}"
        return f"{self.product.name}"

    @property
    def current_price(self):
        """ Returns the active price (sale price if available and lower). """
        price = self.price
        if self.sale_price is not None and self.sale_price < price:
            price = self.sale_price
        return price.quantize(Decimal('0.01'))

    @property
    def total_stock(self):
        """Calculates total stock quantity across all active warehouses."""
        # WARNING: Can cause N+1 queries if used in loops without prefetch_related('stock_levels')
        # Consider calculating this via annotation in querysets where possible
        # Or adding a denormalized field if performance is critical
        # Filters by active warehouses implicitly via Stock model's default manager
        total = self.stock_levels.aggregate(total_quantity=Sum('quantity'))['total_quantity']
        return total or 0


# --- Warehouse Model ---
class Warehouse(SoftDeleteModel): # Inherits soft delete
    name = models.CharField(max_length=255, unique=True)
    # Consider i18n
    name2 = models.CharField(max_length=255, blank=True, null=True, help_text="Russian name (or alternative)")
    slug = models.SlugField(max_length=255, unique=True, blank=True, help_text="Leave blank to auto-generate")
    description = models.TextField(blank=True, null=True)
    type = models.CharField(max_length=50, choices=[('physical', 'Physical'), ('virtual', 'Virtual')], default='physical')
    address = models.ForeignKey(Address, on_delete=models.SET_NULL, null=True, blank=True, related_name='warehouses')
    contact_info = models.CharField(max_length=255, blank=True, null=True)
    is_active = models.BooleanField(default=True, db_index=True, help_text="Is this warehouse operational?") # Renamed from status

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = generate_unique_slug(self, 'name', 'slug')
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

# --- Stock Model (Inventory per Variant per Warehouse) ---
class Stock(SoftDeleteModel): # Inherits soft delete (tracks if a stock record itself is active/deleted)
    warehouse = models.ForeignKey(Warehouse, on_delete=models.CASCADE, related_name='stock_items')
    product_variant = models.ForeignKey(ProductVariant, on_delete=models.CASCADE, related_name='stock_levels')
    quantity = models.IntegerField(default=0, help_text="Current quantity on hand")
    min_quantity = models.PositiveIntegerField(default=0, help_text="Minimum desired quantity (for reordering reports)")
    last_counted_at = models.DateTimeField(null=True, blank=True)

    class Meta(SoftDeleteModel.Meta):
        unique_together = ('warehouse', 'product_variant')
        indexes = [models.Index(fields=['warehouse', 'product_variant'])]
        verbose_name = "Stock Level"
        verbose_name_plural = "Stock Levels"

    def __str__(self):
        variant_name = self.product_variant.display_name if self.product_variant else "N/A"
        warehouse_name = self.warehouse.name if self.warehouse else "N/A"
        return f"{variant_name} ({self.quantity}) in {warehouse_name}"

    # --- Atomic Stock Updates (Including Audit Trail) ---
    # These methods ensure that quantity updates are safe under concurrent requests
    # and create an audit trail via StockMovement.

    @classmethod
    @transaction.atomic
    def increase_quantity(cls, product_variant_id, warehouse_id, amount, reason_type=None, reason_ref=None, user=None, select_for_update=True):
        """
        Atomically increases stock quantity and logs the movement.
        Returns updated quantity or None if stock record doesn't exist.
        """
        if amount <= 0:
            raise ValueError("Amount to increase must be positive.")

        # Use _base_manager if you need to operate on a soft-deleted Stock record
        stock_qs = cls.objects.filter(product_variant_id=product_variant_id, warehouse_id=warehouse_id)
        if select_for_update:
            stock_qs = stock_qs.select_for_update() # Lock the row

        stock = stock_qs.first()
        if not stock:
            logger.warning(f"Attempted to increase stock for non-existent record: variant {product_variant_id}, warehouse {warehouse_id}")
            # Option: Create the stock record if it doesn't exist?
            # stock = cls.objects.create(product_variant_id=product_variant_id, warehouse_id=warehouse_id, quantity=0)
            # if not stock: return None # Handle creation failure
            return None # Or raise error

        updated_count = stock_qs.update(quantity=F('quantity') + amount) # Use filter directly

        if updated_count > 0:
            new_quantity = stock.quantity + amount
            # Log the movement
            StockMovement.objects.create(
                stock=stock,
                product_variant_id=product_variant_id,
                warehouse_id=warehouse_id,
                quantity_change=amount,
                new_quantity=new_quantity,
                reason_type=reason_type,
                reason_reference=str(reason_ref) if reason_ref else None,
                user=user
            )
            return new_quantity
        else:
            logger.error(f"Stock increase failed unexpectedly for variant {product_variant_id}, warehouse {warehouse_id}")
            return None # Or raise an error

    @classmethod
    @transaction.atomic
    def decrease_quantity(cls, product_variant_id, warehouse_id, amount, allow_negative=False, reason_type=None, reason_ref=None, user=None, select_for_update=True):
        """
        Atomically decreases stock quantity and logs the movement.
        Returns updated quantity.
        Raises ValueError if insufficient stock and allow_negative is False.
        Returns None if stock record doesn't exist or update fails.
        """
        if amount <= 0:
            raise ValueError("Amount to decrease must be positive.")

        # Use _base_manager if operating on soft-deleted records is needed
        stock_qs = cls.objects.filter(product_variant_id=product_variant_id, warehouse_id=warehouse_id)
        if select_for_update:
            stock_qs = stock_qs.select_for_update()

        stock = stock_qs.first() # Get the instance to check current quantity

        if not stock:
             logger.warning(f"Attempted to decrease stock for non-existent record: variant {product_variant_id}, warehouse {warehouse_id}")
             return None # Or raise

        if not allow_negative and stock.quantity < amount:
            raise ValueError(f"Insufficient stock for variant {product_variant_id} in warehouse {warehouse_id}. "
                             f"Required: {amount}, Available: {stock.quantity}")

        # Perform the update using F() expression
        updated_count = stock_qs.update(quantity=F('quantity') - amount)

        if updated_count > 0:
            new_quantity = stock.quantity - amount
            # Log the movement
            StockMovement.objects.create(
                stock=stock,
                product_variant_id=product_variant_id,
                warehouse_id=warehouse_id,
                quantity_change=-amount, # Store decrease as negative
                new_quantity=new_quantity,
                reason_type=reason_type,
                reason_reference=str(reason_ref) if reason_ref else None,
                user=user
            )
            return new_quantity
        else:
            # Should not happen if stock existed unless update failed unexpectedly
            logger.error(f"Stock decrease failed unexpectedly for variant {product_variant_id}, warehouse {warehouse_id}")
            return None # Or raise

# --- Stock Movement Model (Audit Trail) ---
class StockMovement(BaseModel): # Does not need soft delete usually
    REASON_CHOICES = [
        ('sale', 'Sale'),
        ('purchase', 'Purchase Order Receipt'),
        ('return', 'Customer Return'),
        ('adjustment', 'Manual Adjustment'),
        ('transfer_out', 'Warehouse Transfer Out'),
        ('transfer_in', 'Warehouse Transfer In'),
        ('initial', 'Initial Stock Count'),
        ('other', 'Other'),
    ]

    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='movements', help_text="The specific stock record affected")
    product_variant = models.ForeignKey(ProductVariant, on_delete=models.PROTECT, related_name='stock_movements', help_text="The variant involved (Protected to keep history)")
    warehouse = models.ForeignKey(Warehouse, on_delete=models.PROTECT, related_name='stock_movements', help_text="The warehouse involved (Protected to keep history)")
    quantity_change = models.IntegerField(help_text="Amount stock changed by (+ve for increase, -ve for decrease)")
    new_quantity = models.IntegerField(help_text="Quantity after this movement occurred")
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    reason_type = models.CharField(max_length=20, choices=REASON_CHOICES, blank=True, null=True, db_index=True)
    reason_reference = models.CharField(max_length=100, blank=True, null=True, help_text="Reference ID (e.g., Sale ID, PO Number, User Note)")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name='stock_adjustments')

    class Meta(BaseModel.Meta):
        verbose_name = "Stock Movement"
        verbose_name_plural = "Stock Movements"
        indexes = [
            models.Index(fields=['stock', 'timestamp']),
            models.Index(fields=['product_variant', 'warehouse', 'timestamp']),
            models.Index(fields=['reason_type', 'timestamp']),
        ]
        ordering = ['-timestamp', '-created_at']

    def __str__(self):
        change_str = f"+{self.quantity_change}" if self.quantity_change > 0 else str(self.quantity_change)
        variant_sku = self.product_variant.sku if self.product_variant else "N/A"
        warehouse_name = self.warehouse.name if self.warehouse else "N/A"
        return f"{change_str} units of {variant_sku} in {warehouse_name} at {self.timestamp.strftime('%Y-%m-%d %H:%M')} ({self.get_reason_type_display() or 'Unknown'})"


# --- Contragent Model (Customer, Supplier, etc.) ---
class Contragent(SoftDeleteModel): # Inherits soft delete
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        null=True, blank=True, # Allow contragents not linked to a system user
        on_delete=models.SET_NULL,
        related_name='contragent_profile'
    )
    name = models.CharField(max_length=255, db_index=True)
    # Consider i18n
    name2 = models.CharField(max_length=255, blank=True, null=True, help_text="Russian name (or alternative)")
    slug = models.SlugField(max_length=255, unique=True, blank=True, help_text="Leave blank to auto-generate")
    phone = models.CharField(max_length=50, blank=True, null=True)
    email = models.EmailField(null=True, blank=True, db_index=True)
    type = models.CharField(max_length=50, choices=[('customer', 'Customer'), ('supplier', 'Supplier'), ('employee', 'Employee'), ('other', 'Other')], default='customer', db_index=True)

    # Link to default addresses
    default_billing_address = models.ForeignKey(Address, on_delete=models.SET_NULL, null=True, blank=True, related_name='contragents_billing')
    default_shipping_address = models.ForeignKey(Address, on_delete=models.SET_NULL, null=True, blank=True, related_name='contragents_shipping')

    is_active = models.BooleanField(default=True, db_index=True) # Renamed from status
    company_name = models.CharField(max_length=255, blank=True, null=True)
    vat_id = models.CharField(max_length=50, blank=True, null=True, db_index=True)

    def save(self, *args, **kwargs):
        if not self.slug and self.name:
            self.slug = generate_unique_slug(self, 'name', 'slug')
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.name} ({self.get_type_display()})"


# --- Sale (Order) Model ---
class Sale(SoftDeleteModel): # Inherits soft delete (allows cancelling/hiding orders)
    SALE_STATUS_CHOICES = [
        ('pending', 'Pending'),         # Order received, awaiting processing/payment
        ('processing', 'Processing'),   # Payment received/confirmed, preparing for shipment
        ('shipped', 'Shipped'),         # Order handed over to carrier
        ('completed', 'Completed'),     # Order delivered/fulfilled
        ('cancelled', 'Cancelled'),     # Order cancelled before completion
        ('refunded', 'Refunded'),       # Full or partial refund processed
        ('payment_failed', 'Payment Failed'), # Payment attempt failed
    ]
    PAYMENT_STATUS_CHOICES = [
        ('unpaid', 'Unpaid'),           # No payment received
        ('paid', 'Paid'),               # Full payment received
        ('partial', 'Partially Paid'),  # Partial payment received
        ('refunded', 'Refunded'),       # Payment refunded (fully or partially)
        ('failed', 'Failed'),           # Payment attempt failed
    ]

    # Consider database sequence or more robust generator for high concurrency
    # See comment in original feedback about potential race condition with uuid+timestamp
    sale_number = models.CharField(max_length=50, unique=True, blank=True, db_index=True, help_text="Unique identifier for the sale (auto-generated)")
    customer = models.ForeignKey(Contragent, on_delete=models.PROTECT, related_name='sales', limit_choices_to={'type': 'customer'}, db_index=True)
    sale_date = models.DateTimeField(default=timezone.now, db_index=True)
    status = models.CharField(max_length=20, choices=SALE_STATUS_CHOICES, default='pending', db_index=True)

    # --- Address Snapshotting ---
    # These FKs should point to *clones* of the Address objects created at the time of sale.
    # The cloning logic needs to be implemented in the view/service layer that creates the Sale.
    shipping_address = models.ForeignKey(
        Address,
        on_delete=models.SET_NULL, # Keep sale history even if snapshot address is deleted (though unlikely)
        null=True, blank=True,
        related_name='shipping_sales',
        help_text="Snapshot of the shipping address used for this sale (should be a clone)"
    )
    billing_address = models.ForeignKey(
        Address,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='billing_sales',
        help_text="Snapshot of the billing address used for this sale (should be a clone)"
    )

    # Financials - Totals
    discount_amount = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal('0.00'), help_text="Order-level discount amount")
    shipping_cost = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal('0.00'), help_text="Cost of shipping for this order")

    # Payment details
    payment_method = models.CharField(max_length=50, blank=True, null=True)
    payment_status = models.CharField(max_length=20, choices=PAYMENT_STATUS_CHOICES, default='unpaid', db_index=True)
    transaction_id = models.CharField(max_length=100, blank=True, null=True, db_index=True, help_text="Payment gateway transaction ID")

    # Deductions (Percentages stored, calculated amounts via properties)
    tax_percentage = models.DecimalField(max_digits=5, decimal_places=2, default=Decimal('0.00'), help_text="e.g., Sales tax percentage (3.00 for 3%) applied to total_amount")
    acquiring_percentage = models.DecimalField(max_digits=5, decimal_places=2, default=Decimal('0.00'), help_text="e.g., Payment gateway fee percentage (0.95 for 0.95%) applied to total_amount")
    commission_percentage = models.DecimalField(max_digits=5, decimal_places=2, default=Decimal('0.00'), help_text="e.g., Salesperson commission percentage (10.00 for 10%) applied to total_amount")

    notes = models.TextField(blank=True, null=True, help_text="Internal notes or customer comments")

    class Meta(SoftDeleteModel.Meta):
        verbose_name = "Sale / Order"
        verbose_name_plural = "Sales / Orders"
        indexes = [
            models.Index(fields=['customer', 'sale_date']),
            models.Index(fields=['status']),
            models.Index(fields=['payment_status']),
            models.Index(fields=['transaction_id']), # Index if searching by this often
            # sale_number has unique=True which implies an index
        ]
        ordering = ['-sale_date', '-created_at']

    def save(self, *args, **kwargs):
        is_new = self._state.adding
        if is_new and not self.sale_number:
            # Generate unique sale number *before* first save
            timestamp = timezone.now().strftime('%Y%m%d%H%M%S')
            unique_part = uuid.uuid4().hex[:8].upper()
            potential_number = f"S-{timestamp}-{unique_part}"
            # Use _base_manager to check existence across all, including soft-deleted
            while Sale.all_objects.filter(sale_number=potential_number).exists():
                unique_part = uuid.uuid4().hex[:8].upper()
                potential_number = f"S-{timestamp}-{unique_part}"
            self.sale_number = potential_number

        # Address assignment/cloning should happen *before* calling save in the view/service
        super().save(*args, **kwargs)

    # --- Calculated Properties ---
    # WARNING: These can cause N+1 queries. Use prefetch_related('items') on Sale querysets.
    @property
    def subtotal(self):
        """Calculates the subtotal from all sale items *after* item-level discounts."""
        total = sum(item.line_total for item in self.items.all() if item.line_total is not None)
        return total.quantize(Decimal('0.01')) if total is not None else Decimal('0.00')

    @property
    def total_amount(self):
        """Calculates the final amount paid by the customer (subtotal - order discounts + shipping)."""
        return (self.subtotal - self.discount_amount + self.shipping_cost).quantize(Decimal('0.01'))

    @property
    def tax_deduction(self):
        """Calculates the tax amount based on the total amount."""
        if self.tax_percentage > 0:
            return (self.total_amount * (self.tax_percentage / Decimal('100'))).quantize(Decimal('0.01'))
        return Decimal('0.00')

    @property
    def acquiring_deduction(self):
        """Calculates the acquiring fee based on the total amount."""
        if self.acquiring_percentage > 0:
            return (self.total_amount * (self.acquiring_percentage / Decimal('100'))).quantize(Decimal('0.01'))
        return Decimal('0.00')

    @property
    def commission_deduction(self):
        """Calculates the commission based on the total amount."""
        if self.commission_percentage > 0:
            return (self.total_amount * (self.commission_percentage / Decimal('100'))).quantize(Decimal('0.01'))
        return Decimal('0.00')

    @property
    def total_deductions(self):
        """Calculates the sum of all seller-side deductions."""
        return (self.tax_deduction + self.acquiring_deduction + self.commission_deduction).quantize(Decimal('0.01'))

    @property
    def net_revenue(self):
        """Calculates the revenue remaining after all deductions."""
        return (self.total_amount - self.total_deductions).quantize(Decimal('0.01'))

    def __str__(self):
        customer_name = self.customer.name if self.customer else "N/A"
        return f"Sale {self.sale_number} ({customer_name})"


# --- Sale Item Model (Line Item within a Sale) ---
class SaleItem(SoftDeleteModel): # Inherits soft delete (allows removing item from cancelled order etc.)
    sale = models.ForeignKey(Sale, on_delete=models.CASCADE, related_name='items', db_index=True)
    # Link to the specific variant sold
    product_variant = models.ForeignKey(ProductVariant, on_delete=models.PROTECT, related_name='sale_items') # PROTECT to prevent deleting variants that were sold
    quantity = models.PositiveIntegerField(default=1)

    # --- Snapshot Details ---
    # Capture details AT THE TIME OF SALE
    product_name = models.CharField(max_length=300, blank=True, help_text="Product name (incl. variant) at time of sale")
    sku_at_sale = models.CharField(max_length=100, blank=True, null=True, db_index=True, help_text="Variant SKU at time of sale")
    price_at_sale = models.DecimalField(max_digits=12, decimal_places=2, help_text="Price per unit at time of sale (before item discount)")
    # Item-level discount (total discount for the line, not per unit)
    discount_amount = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal('0.00'), help_text="Total discount applied to this line item")

    class Meta(SoftDeleteModel.Meta):
        unique_together = ('sale', 'product_variant') # Usually one line per variant per sale
        verbose_name = "Sale Item"
        verbose_name_plural = "Sale Items"
        ordering = ['created_at'] # Order items as added

    def save(self, *args, **kwargs):
        # Capture product/variant details only if they are not explicitly set during creation
        # It's STRONGLY recommended to set these explicitly when creating the SaleItem instance.
        if not self.pk: # Only set defaults on creation if not provided
            if self.product_variant and not self.product_name:
                self.product_name = self.product_variant.display_name[:300] # Truncate
            if self.product_variant and not self.sku_at_sale:
                self.sku_at_sale = self.product_variant.sku
            if self.product_variant and self.price_at_sale is None:
                # Critical: Price MUST be determined based on rules at time of sale creation
                logger.error(f"SaleItem for variant {self.product_variant.sku} in sale {self.sale_id} is being saved without explicit price_at_sale!")
                # Fallback is dangerous as current_price might change later
                self.price_at_sale = self.product_variant.current_price # Use as last resort fallback

        super().save(*args, **kwargs)
        # Note: If Sale totals were denormalized fields, trigger sale total update here (e.g., via signals or direct call)

    @property
    def line_subtotal(self):
         """Calculates the subtotal for this line item (quantity * price) before item discount."""
         if self.price_at_sale is not None and self.quantity is not None:
             return (self.quantity * self.price_at_sale).quantize(Decimal('0.01'))
         return Decimal('0.00')

    @property
    def line_total(self):
        """Calculates the final total for this line item (subtotal - item discount)."""
        # Assuming discount_amount is the TOTAL discount for the line
        return (self.line_subtotal - self.discount_amount).quantize(Decimal('0.01'))

    def __str__(self):
        p_name = self.product_name or (self.product_variant.display_name if self.product_variant else "N/A")
        sale_num = self.sale.sale_number if self.sale_id else 'N/A'
        return f"{self.quantity} x {p_name} in Sale {sale_num}"


# --- Shopping Cart Models ---

class Cart(BaseModel): # Carts are typically ephemeral, maybe no soft delete needed?
    """ Represents a shopping cart """
    # Link to user or use session key for anonymous users
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True, blank=True,
        on_delete=models.CASCADE, # Delete cart if user is deleted
        related_name='carts'
    )
    # Store session key for anonymous users
    session_key = models.CharField(max_length=40, null=True, blank=True, db_index=True)
    # Could add status ('active', 'merged', 'ordered', 'abandoned')

    class Meta(BaseModel.Meta):
        # Ensure a user doesn't have multiple active carts? Or allow it?
        # unique_together = ('user', 'session_key') # Might need refinement based on logic
        verbose_name = "Shopping Cart"
        verbose_name_plural = "Shopping Carts"

    def __str__(self):
        if self.user:
            return f"Cart for {self.user.username}"
        elif self.session_key:
            return f"Anonymous Cart ({self.session_key})"
        return f"Cart {self.id}"

    @property
    def total_items(self):
        """ Returns the total number of individual items in the cart. """
        # WARNING: Potential N+1 query. Use prefetch_related('items') or aggregate.
        # return sum(item.quantity for item in self.items.all())
        result = self.items.aggregate(total_quantity=Sum('quantity'))
        return result['total_quantity'] or 0

    @property
    def total_price(self):
        """ Calculates the total price of all items in the cart using current variant prices. """
        # WARNING: Potential N+1 query. Use prefetch_related('items__product_variant')
        total = Decimal('0.00')
        # Use select_related('product_variant') for efficiency within the loop
        for item in self.items.select_related('product_variant'):
            total += item.line_total
        return total.quantize(Decimal('0.01'))

class CartItem(BaseModel): # Also likely ephemeral
    """ Represents an item within a shopping cart """
    cart = models.ForeignKey(Cart, on_delete=models.CASCADE, related_name='items')
    product_variant = models.ForeignKey(ProductVariant, on_delete=models.CASCADE, related_name='cart_items') # Cascade delete if variant removed
    quantity = models.PositiveIntegerField(default=1)
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta(BaseModel.Meta):
        unique_together = ('cart', 'product_variant') # Only one line item per variant per cart
        verbose_name = "Cart Item"
        verbose_name_plural = "Cart Items"
        ordering = ['added_at']

    @property
    def unit_price(self):
        """ Gets the current price of the associated product variant. """
        # WARNING: Potential N+1 query if used in loops without select_related('product_variant')
        if self.product_variant:
            return self.product_variant.current_price
        return Decimal('0.00')

    @property
    def line_total(self):
        """ Calculates the total price for this line item based on current price. """
        return (self.quantity * self.unit_price).quantize(Decimal('0.01'))

    def __str__(self):
        variant_name = self.product_variant.display_name if self.product_variant else "N/A"
        return f"{self.quantity} x {variant_name} in Cart {self.cart_id}"


# --- END OF FILE models.py ---