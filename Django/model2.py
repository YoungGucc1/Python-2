

import uuid
from django.db import models
from django.contrib.auth.models import User
from django.utils.text import slugify
from django.utils import timezone # Import timezone
from PIL import Image as PILImage
import os
from decimal import Decimal # Import Decimal for calculations

# Abstract base model for common fields
class BaseModel(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)
    is_deleted = models.BooleanField(default=False)  # Soft delete flag

    class Meta:
        abstract = True
        ordering = ['-created_at'] # Default ordering

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
        update_meta = False
        if self._state.adding: # Check if it's a new instance before super().save()
             update_meta = True
        elif self.file_path and 'file_path' in kwargs.get('update_fields', []): # Check if file_path is being updated
             update_meta = True

        super().save(*args, **kwargs)

        if update_meta and self.file_path:
            try:
                img = PILImage.open(self.file_path.path)
                # Optimize image
                max_size = (1920, 1080)  # Max dimensions for optimization
                img_changed = False
                if img.width > max_size[0] or img.height > max_size[1]:
                    img.thumbnail(max_size, PILImage.Resampling.LANCZOS)
                    img_changed = True

                if img_changed:
                    img.save(self.file_path.path, quality=85, optimize=True)  # Reduce quality and optimize

                # Update metadata
                self.resolution = f"{img.width}x{img.height}"
                self.format = img.format
                self.size = os.path.getsize(self.file_path.path) // 1024  # Size in KB

                # Save updated fields without triggering infinite loop
                super().save(update_fields=['resolution', 'size', 'format'])

            except FileNotFoundError:
                # Handle case where file path might not exist immediately after save
                # (e.g., remote storage) - You might need more robust handling here.
                print(f"Warning: Could not process image {self.file_path.path}, file not found.")
            except Exception as e:
                print(f"Error processing image {self.file_path.path}: {e}")


    def __str__(self):
        if self.file_path:
            return f"{os.path.basename(self.file_path.name)} ({self.format})"
        return f"Image {self.id}"

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
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, related_name='children', db_index=True)
    image = models.ForeignKey(Image, null=True, blank=True, on_delete=models.SET_NULL)

    # SEO fields
    meta_title = models.CharField(max_length=255, blank=True, null=True)
    meta_description = models.TextField(blank=True, null=True)

    class Meta(BaseModel.Meta): # Inherit Meta options if needed
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
    name = models.CharField(max_length=255, db_index=True) # Index name for lookups
    name2 = models.CharField(max_length=255, blank=True, null=True)
    slug = models.SlugField(max_length=255, unique=True, blank=True)
    description = models.TextField(blank=True, null=True)
    short_description = models.CharField(max_length=255, blank=True, null=True)
    category = models.ForeignKey(Category, on_delete=models.PROTECT, db_index=True, related_name='products') # Use PROTECT
    brand = models.ForeignKey(Brand, on_delete=models.PROTECT, null=True, blank=True, related_name='products') # Use PROTECT
    images = models.ManyToManyField(Image, related_name='products', blank=True) # Allow blank
    is_featured = models.BooleanField(default=False, db_index=True)
    is_active = models.BooleanField(default=True, db_index=True)
    sku = models.CharField(max_length=100, unique=True, blank=True, null=True) # Allow blank/null SKU temporarily? Or enforce?
    weight = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    sale_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)

    # SEO fields
    meta_title = models.CharField(max_length=255, blank=True, null=True)
    meta_description = models.TextField(blank=True, null=True)

    # Tags implementation - Reconsider using Brand for tags? Maybe a dedicated Tag model?
    # Using Brand here seems semantically incorrect unless Brands act as tags.
    # If you want generic tags, create a Tag model. Let's remove this for now.
    # tags = models.ManyToManyField(Brand, related_name='tagged_products', blank=True)

    class Meta(BaseModel.Meta): # Inherit Meta options
        indexes = [
            models.Index(fields=['slug']),
            models.Index(fields=['is_active', 'is_featured']),
            models.Index(fields=['name']), # Index name if frequently searched
            # models.Index(fields=['sku']), # Index SKU if frequently searched
        ]
        # Consider making name unique or unique together with brand/category?
        # unique_together = (('name', 'brand'),) # Example

    def save(self, *args, **kwargs):
        if not self.slug:
            # Create a more robust slug, e.g., including brand or sku if name not unique
            base_slug = slugify(self.name)
            self.slug = base_slug
            # Add logic here if needed to ensure slug uniqueness, e.g., append ID or count
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

    @property
    def current_price(self):
        """Returns the active price (sale price if available and valid)."""
        if self.sale_price is not None and self.sale_price < self.price:
             # Add logic here if sale price has start/end dates
             return self.sale_price
        return self.price

# Warehouse Model
class Warehouse(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    name2 = models.CharField(max_length=255, blank=True, null=True)
    slug = models.SlugField(max_length=255, unique=True, blank=True)
    description = models.TextField(blank=True, null=True)
    type = models.CharField(max_length=50, choices=[('physical', 'Physical'), ('virtual', 'Virtual')], default='physical')
    address = models.TextField(blank=True, null=True) # Allow blank address?
    contact_info = models.CharField(max_length=255, blank=True, null=True)
    status = models.CharField(max_length=20, choices=[('active', 'Active'), ('inactive', 'Inactive')], default='active')

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

# Stock Model
class Stock(BaseModel):
    warehouse = models.ForeignKey(Warehouse, on_delete=models.CASCADE, related_name='stock_items') # Renamed related_name
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='stock_levels') # Renamed related_name
    quantity = models.IntegerField(default=0) # Use IntegerField, allows negative if needed for adjustments
    min_quantity = models.PositiveIntegerField(default=0)
    last_counted_at = models.DateTimeField(null=True, blank=True)

    class Meta(BaseModel.Meta): # Inherit Meta options
        unique_together = ('warehouse', 'product')
        indexes = [models.Index(fields=['warehouse', 'product'])]

    def __str__(self):
        return f"{self.product.name} ({self.quantity}) in {self.warehouse.name}"

# Contragent Model (Consider renaming to Customer or Contact if more appropriate)
class Contragent(BaseModel):
    # Link to User is optional - a contragent might not be a website user
    user = models.OneToOneField(User, null=True, blank=True, on_delete=models.SET_NULL, related_name='contragent_profile') # Changed related_name
    name = models.CharField(max_length=255, db_index=True) # Don't enforce unique name, people can share names
    name2 = models.CharField(max_length=255, blank=True, null=True)
    # Slug might not be needed/useful for Contragents unless they have public profiles
    slug = models.SlugField(max_length=255, unique=True, blank=True)
    phone = models.CharField(max_length=50, blank=True, null=True)
    email = models.EmailField(unique=True, null=True, blank=True) # Allow null/blank email, maybe not unique?
    type = models.CharField(max_length=50, choices=[('customer', 'Customer'), ('supplier', 'Supplier'), ('employee', 'Employee'), ('other', 'Other')], default='customer') # Added more types
    address = models.TextField(blank=True, null=True) # Billing/default address
    status = models.CharField(max_length=20, choices=[('active', 'Active'), ('inactive', 'Inactive')], default='active')
    # Add company details if applicable (VAT ID, Company Name etc.)
    company_name = models.CharField(max_length=255, blank=True, null=True)
    vat_id = models.CharField(max_length=50, blank=True, null=True)

    def save(self, *args, **kwargs):
        if not self.slug and self.name: # Only create slug if name exists
             # Create a potentially non-unique slug, or add logic for uniqueness if needed
             self.slug = slugify(self.name)
             # Consider adding UUID or ID to slug for uniqueness if required
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.name} ({self.get_type_display()})"


# ----- NEW MODELS START -----

class Sale(BaseModel):
    """ Represents a single sales transaction/order. """
    SALE_STATUS_CHOICES = [
        ('pending', 'Pending'),          # Order placed, awaiting payment/processing
        ('processing', 'Processing'),    # Payment received, preparing for shipment
        ('shipped', 'Shipped'),          # Order shipped
        ('completed', 'Completed'),      # Order received by customer (or final state)
        ('cancelled', 'Cancelled'),      # Order cancelled
        ('refunded', 'Refunded'),        # Order refunded
    ]

    sale_number = models.CharField(max_length=50, unique=True, blank=True, help_text="Unique identifier for the sale (e.g., order number)")
    customer = models.ForeignKey(Contragent, on_delete=models.PROTECT, related_name='sales', limit_choices_to={'type': 'customer'}, db_index=True)
    sale_date = models.DateTimeField(default=timezone.now, db_index=True)
    status = models.CharField(max_length=20, choices=SALE_STATUS_CHOICES, default='pending', db_index=True)

    # Store addresses at the time of sale, as customer's default might change
    shipping_address = models.TextField(blank=True, null=True)
    billing_address = models.TextField(blank=True, null=True)

    # Financials
    # subtotal = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal('0.00')) # Calculated from items
    discount_amount = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal('0.00'))
    # shipping_cost = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0.00'))
    # tax_amount = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0.00')) # If tracking tax
    # total_amount = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal('0.00')) # Calculated overall total

    # Payment details (optional, depending on requirements)
    payment_method = models.CharField(max_length=50, blank=True, null=True)
    payment_status = models.CharField(max_length=20, choices=[('unpaid', 'Unpaid'), ('paid', 'Paid'), ('partial', 'Partially Paid')], default='unpaid')
    transaction_id = models.CharField(max_length=100, blank=True, null=True, help_text="Payment gateway transaction ID")

    notes = models.TextField(blank=True, null=True, help_text="Internal notes or customer comments")

    class Meta(BaseModel.Meta): # Inherit Meta options
        verbose_name = "Sale / Order"
        verbose_name_plural = "Sales / Orders"
        indexes = [
            models.Index(fields=['sale_number']),
            models.Index(fields=['customer', 'sale_date']),
        ]

    def generate_sale_number(self):
        """Generates a unique sale number (example implementation)."""
        # Example: Use timestamp + part of UUID
        timestamp = timezone.now().strftime('%Y%m%d%H%M%S')
        unique_part = str(self.id).split('-')[0] # Use first part of UUID
        return f"SALE-{timestamp}-{unique_part.upper()}"

    def save(self, *args, **kwargs):
        if not self.sale_number:
            # Generate sale number just before the first save
            # We need the ID first if using it in the number, so save, generate, save again.
            # Or generate without ID initially. Let's generate without ID for simplicity here.
            # A more robust solution might use signals or a separate sequence generator.
             timestamp = timezone.now().strftime('%Y%m%d')
             # Find count of sales today for sequence (simplistic)
             today_min = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
             today_max = timezone.now().replace(hour=23, minute=59, second=59, microsecond=999999)
             todays_sales_count = Sale.objects.filter(sale_date__range=(today_min, today_max)).count()
             self.sale_number = f"S{timestamp}-{todays_sales_count + 1:04d}"
             # Ensure uniqueness loop (optional, add if collisions are likely)

        # Optionally copy addresses from customer if not provided
        if not self.billing_address and self.customer:
             self.billing_address = self.customer.address
        # Add logic for shipping address if needed (e.g., separate shipping address model or field in Contragent)

        # Calculate totals (better done with properties or methods to avoid stale data)
        # self.update_totals() # Call calculation method if storing totals

        super().save(*args, **kwargs)

    @property
    def subtotal(self):
        """Calculates the subtotal from all sale items."""
        return sum(item.line_total for item in self.items.all())

    @property
    def total_amount(self):
        """Calculates the final total amount after discounts."""
        # Add shipping, tax etc. here if applicable
        return self.subtotal - self.discount_amount

    # Example method to update stored totals (if you choose to store them)
    # def update_totals(self):
    #     self.subtotal = self.calculate_subtotal()
    #     self.total_amount = self.subtotal - self.discount_amount # + self.shipping_cost + self.tax_amount
    #     # Note: Saving here inside update_totals can cause recursion if called from save().
    #     # Better to calculate in save() or use properties.

    def __str__(self):
        return f"Sale {self.sale_number} ({self.customer.name})"


class SaleItem(BaseModel):
    """ Represents a single product line item within a Sale. """
    sale = models.ForeignKey(Sale, on_delete=models.CASCADE, related_name='items', db_index=True)
    product = models.ForeignKey(Product, on_delete=models.PROTECT, related_name='sale_items') # Protect product from deletion if sold
    # Optional: Link to specific variant if you have product variants
    # product_variant = models.ForeignKey(ProductVariant, on_delete=models.PROTECT, null=True, blank=True)

    quantity = models.PositiveIntegerField(default=1)
    # Store price/name details AT THE TIME OF SALE, as product details might change
    product_name = models.CharField(max_length=255, blank=True, help_text="Product name at time of sale")
    sku_at_sale = models.CharField(max_length=100, blank=True, null=True, help_text="Product SKU at time of sale")
    price_at_sale = models.DecimalField(max_digits=10, decimal_places=2, help_text="Price per unit at time of sale")
    # line_total = models.DecimalField(max_digits=12, decimal_places=2, editable=False) # Calculated

    class Meta(BaseModel.Meta): # Inherit Meta options
        unique_together = ('sale', 'product') # Allow same product once per sale (adjust if variants used)
        verbose_name = "Sale Item"
        verbose_name_plural = "Sale Items"

    def save(self, *args, **kwargs):
        # Capture product details at the time of sale if not already set
        if self.product and not self.product_name:
            self.product_name = self.product.name
        if self.product and not self.sku_at_sale:
            self.sku_at_sale = self.product.sku
        # if not self.price_at_sale: # Should be set when item is added
        #    self.price_at_sale = self.product.current_price # Or specific price logic

        # Calculate line total
        # self.line_total = self.quantity * self.price_at_sale

        super().save(*args, **kwargs)
        # Note: If you store totals on the Sale model, you might need to update
        # the parent Sale totals here using signals or overriding the Sale's save method carefully.
        # Example: self.sale.save() # Could trigger Sale's update_totals

    @property
    def line_total(self):
        """Calculates the total for this line item."""
        if self.price_at_sale is not None:
            return self.quantity * self.price_at_sale
        return Decimal('0.00')

    def __str__(self):
        return f"{self.quantity} x {self.product_name or self.product.name} in Sale {self.sale.sale_number}"


# ----- NEW MODELS END -----


