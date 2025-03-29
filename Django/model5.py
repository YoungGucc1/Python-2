# -*- coding: utf-8 -*-
import uuid
from django.db import models
from django.contrib.auth.models import User
from django.utils.text import slugify
from PIL import Image as PILImage
import os
from decimal import Decimal # Импортируем Decimal для точных расчетов

# Abstract base model for common fields
class BaseModel(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Создано")
    modified_at = models.DateTimeField(auto_now=True, verbose_name="Изменено")
    is_deleted = models.BooleanField(default=False, verbose_name="Удалено (мягкое)")  # Soft delete flag

    class Meta:
        abstract = True
        ordering = ['-created_at'] # По умолчанию сортируем по дате создания

# User Profile extension
class UserProfile(BaseModel):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile', verbose_name="Пользователь")
    phone = models.CharField("Телефон", max_length=50, blank=True, null=True)
    address = models.TextField("Адрес", blank=True, null=True)
    profile_image = models.ImageField("Изображение профиля", upload_to='profile_images/', null=True, blank=True)

    def __str__(self):
        return f"Профиль для {self.user.username}"

    class Meta:
        verbose_name = "Профиль пользователя"
        verbose_name_plural = "Профили пользователей"

# Images Model
class Image(BaseModel):
    file_path = models.ImageField("Файл изображения", upload_to='images/')
    description = models.TextField("Описание", blank=True, null=True)
    resolution = models.CharField("Разрешение", max_length=50, blank=True, null=True, editable=False)
    size = models.PositiveIntegerField("Размер (КБ)", help_text="Размер в килобайтах", blank=True, null=True, editable=False)
    format = models.CharField("Формат", max_length=10, blank=True, null=True, editable=False)
    alt_text = models.CharField("Alt текст (для доступности)", max_length=255, blank=True, null=True)

    def save(self, *args, **kwargs):
        # Сохраняем один раз, чтобы получить путь к файлу, если это новый объект
        is_new = self.pk is None
        super().save(*args, **kwargs)

        update_fields = []
        try:
            # Открываем изображение
            img = PILImage.open(self.file_path.path)

            # Оптимизация (если нужно)
            max_size = (1920, 1080)
            needs_resave = False
            if img.width > max_size[0] or img.height > max_size[1]:
                img.thumbnail(max_size, PILImage.Resampling.LANCZOS)
                needs_resave = True

            # Сжимаем изображение (даже если размер не менялся, для уменьшения веса)
            img.save(self.file_path.path, quality=85, optimize=True)

            # Обновляем метаданные
            current_resolution = f"{img.width}x{img.height}"
            current_format = img.format
            current_size_kb = os.path.getsize(self.file_path.path) // 1024

            if self.resolution != current_resolution:
                self.resolution = current_resolution
                update_fields.append('resolution')
            if self.format != current_format:
                self.format = current_format
                update_fields.append('format')
            if self.size != current_size_kb:
                self.size = current_size_kb
                update_fields.append('size')

            # Закрываем файл изображения
            img.close()

        except FileNotFoundError:
             # Обработка случая, если файл не найден (например, после удаления вручную)
             pass
        except Exception as e:
             # Логирование или обработка других ошибок Pillow/OS
             print(f"Error processing image {self.file_path.path}: {e}")
             # Можно решить не прерывать сохранение из-за ошибки обработки изображения

        # Сохраняем обновленные поля, если они изменились
        if update_fields and not is_new: # Обновляем только если это не первое сохранение
            super().save(update_fields=update_fields)
        elif is_new and update_fields: # Если это первое сохранение, просто обновим объект в памяти
             pass # Уже обновлено выше


    def __str__(self):
        if self.file_path:
            return f"{os.path.basename(self.file_path.name)} ({self.format or 'N/A'})"
        return f"Image {self.id}"

    class Meta:
        verbose_name = "Изображение"
        verbose_name_plural = "Изображения"

# Brand Model
class Brand(BaseModel):
    name = models.CharField("Название", max_length=255, unique=True)
    name2 = models.CharField("Название 2", max_length=255, blank=True, null=True) # Доп. название (например, на другом языке)
    slug = models.SlugField("Slug (URL)", max_length=255, unique=True, blank=True, help_text="Автоматически генерируется из названия, если пустое")
    description = models.TextField("Описание", blank=True, null=True)
    logo = models.ForeignKey(Image, verbose_name="Логотип", null=True, blank=True, on_delete=models.SET_NULL, related_name='brand_logos')
    website = models.URLField("Вебсайт", blank=True, null=True)

    # SEO fields
    meta_title = models.CharField("Meta Title (SEO)", max_length=255, blank=True, null=True)
    meta_description = models.TextField("Meta Description (SEO)", blank=True, null=True)

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Бренд"
        verbose_name_plural = "Бренды"

# Categories Model with hierarchy support
class Category(BaseModel):
    name = models.CharField("Название", max_length=255, unique=True)
    name2 = models.CharField("Название 2", max_length=255, blank=True, null=True)
    slug = models.SlugField("Slug (URL)", max_length=255, unique=True, blank=True, help_text="Автоматически генерируется из названия, если пустое")
    description = models.TextField("Описание", blank=True, null=True)
    parent = models.ForeignKey('self', verbose_name="Родительская категория", null=True, blank=True, on_delete=models.SET_NULL, related_name='children')
    image = models.ForeignKey(Image, verbose_name="Изображение", null=True, blank=True, on_delete=models.SET_NULL)

    # SEO fields
    meta_title = models.CharField("Meta Title (SEO)", max_length=255, blank=True, null=True)
    meta_description = models.TextField("Meta Description (SEO)", blank=True, null=True)

    class Meta:
        verbose_name = "Категория"
        verbose_name_plural = "Категории"
        indexes = [models.Index(fields=['slug'])]

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        # Отображение иерархии в строковом представлении (для админки)
        full_path = [self.name]
        k = self.parent
        while k is not None:
            full_path.append(k.name)
            k = k.parent
        return ' -> '.join(full_path[::-1])


# --- NEW: Deduction Type Model ---
class DeductionType(BaseModel):
    """Типы вычетов, применяемых к цене продажи."""
    CALCULATION_METHODS = (
        ('percentage', 'Процент от цены продажи'),
        ('fixed_amount', 'Фиксированная сумма'),
    )
    name = models.CharField("Название вычета", max_length=100, unique=True, help_text='Например: "УСН 3%", "Эквайринг", "Комиссия продавца"')
    slug = models.SlugField("Slug", max_length=100, unique=True, blank=True)
    description = models.TextField("Описание", blank=True, null=True)
    calculation_method = models.CharField("Метод расчета", max_length=20, choices=CALCULATION_METHODS)
    rate = models.DecimalField(
        "Ставка (%)",
        max_digits=5, decimal_places=2, null=True, blank=True,
        help_text="Процентная ставка (например, 3.00 для 3%). Используется, если метод расчета - 'Процент'."
    )
    fixed_amount = models.DecimalField(
        "Фиксированная сумма",
        max_digits=10, decimal_places=2, null=True, blank=True,
        help_text="Фиксированная сумма вычета. Используется, если метод расчета - 'Фиксированная сумма'."
    )
    is_active = models.BooleanField("Активен", default=True, help_text="Используется ли этот тип вычета в расчетах")

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        if self.calculation_method == 'percentage' and self.rate is not None:
            return f"{self.name} ({self.rate}%)"
        elif self.calculation_method == 'fixed_amount' and self.fixed_amount is not None:
             return f"{self.name} ({self.fixed_amount} руб.)" # Предполагаем рубли
        return self.name

    class Meta:
        verbose_name = "Тип вычета"
        verbose_name_plural = "Типы вычетов"

# Product Model
class Product(BaseModel):
    name = models.CharField("Название", max_length=255, db_index=True) # Индекс по названию для поиска
    name2 = models.CharField("Название 2", max_length=255, blank=True, null=True)
    slug = models.SlugField("Slug (URL)", max_length=255, unique=True, blank=True, help_text="Автоматически генерируется из названия, если пустое")
    description = models.TextField("Полное описание", blank=True, null=True)
    short_description = models.CharField("Краткое описание", max_length=255, blank=True, null=True)
    category = models.ForeignKey(Category, verbose_name="Категория", on_delete=models.PROTECT, db_index=True, related_name='products') # PROTECT вместо CASCADE
    brand = models.ForeignKey(Brand, verbose_name="Бренд", on_delete=models.SET_NULL, null=True, blank=True, related_name='products')
    images = models.ManyToManyField(Image, verbose_name="Изображения", related_name='products', blank=True) # blank=True
    is_featured = models.BooleanField("Рекомендуемый", default=False, db_index=True)
    is_active = models.BooleanField("Активен", default=True, db_index=True, help_text="Отображается ли товар на сайте/в системе")
    sku = models.CharField("Артикул (SKU)", max_length=100, unique=True, db_index=True)
    weight = models.DecimalField("Вес (кг)", max_digits=10, decimal_places=3, null=True, blank=True) # 3 знака для граммов

    # --- Цены ---
    # Убираем price и sale_price отсюда, будем использовать модель Price
    # price = models.DecimalField("Базовая цена", max_digits=10, decimal_places=2)
    # sale_price = models.DecimalField("Цена со скидкой", max_digits=10, decimal_places=2, null=True, blank=True)
    cost_price = models.DecimalField("Себестоимость", max_digits=10, decimal_places=2, null=True, blank=True, help_text="Закупочная стоимость товара")

    # --- Вычеты при продаже ---
    applicable_deductions = models.ManyToManyField(
        DeductionType,
        verbose_name="Применимые вычеты при продаже",
        blank=True,
        related_name='products',
        help_text="Какие стандартные вычеты применяются при продаже этого товара"
    )

    # SEO fields
    meta_title = models.CharField("Meta Title (SEO)", max_length=255, blank=True, null=True)
    meta_description = models.TextField("Meta Description (SEO)", blank=True, null=True)

    # Tags implementation (используем Brand как теги - если это задумка)
    tags = models.ManyToManyField(Brand, related_name='tagged_products', blank=True, verbose_name="Теги (Бренды)")

    class Meta:
        verbose_name = "Товар"
        verbose_name_plural = "Товары"
        indexes = [
            models.Index(fields=['slug']),
            models.Index(fields=['is_active', 'is_featured']),
            models.Index(fields=['sku']),
            models.Index(fields=['name']), # Добавим индекс на имя для быстрого поиска
        ]
        # Убрали unique=True с name, т.к. товары с одинаковым названием могут быть (но с разным SKU)
        # unique_together = (('name', 'brand'),) # Можно добавить, если имя+бренд должны быть уникальны

    def save(self, *args, **kwargs):
        if not self.slug:
            # Генерируем slug из имени и SKU для большей уникальности, если нужно
            base_slug = slugify(self.name)
            self.slug = f"{base_slug}-{self.sku or uuid.uuid4().hex[:6]}" # Добавляем SKU или часть UUID
            # Нужна логика для проверки уникальности slug перед сохранением, если требуется
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.name} ({self.sku})"

    # --- Методы для получения цен (пример) ---
    def get_retail_price(self):
        """Получить текущую розничную цену."""
        try:
            # Ищем цену с типом 'retail' (или как вы назовете слаг розничной цены)
            price_entry = self.prices.get(price_type__slug='retail', is_active=True)
            return price_entry.value
        except Price.DoesNotExist:
            return None # Или вернуть 0, или ошибку
        except Price.MultipleObjectsReturned:
             # Обработать случай, когда найдено несколько активных розничных цен (ошибка данных)
             # Например, вернуть самую последнюю
            price_entry = self.prices.filter(price_type__slug='retail', is_active=True).order_by('-created_at').first()
            return price_entry.value if price_entry else None

    def get_effective_selling_price(self):
        """Получить актуальную цену продажи (розничную или со скидкой, если есть)."""
        retail_price = self.get_retail_price()
        # Логика для определения цены со скидкой (может быть отдельный тип цены "sale")
        # sale_price = self.get_sale_price()
        # if sale_price and sale_price < retail_price:
        #    return sale_price
        return retail_price # Пока просто возвращаем розничную

    def calculate_net_proceeds(self, selling_price=None):
        """
        Рассчитать чистую выручку после вычетов (без учета себестоимости).
        selling_price: Цена, по которой товар был фактически продан.
                       Если не указана, используется get_effective_selling_price().
        """
        if selling_price is None:
            selling_price = self.get_effective_selling_price()

        if selling_price is None or selling_price <= 0:
            return Decimal('0.00')

        total_deduction = Decimal('0.00')
        for deduction in self.applicable_deductions.filter(is_active=True):
            if deduction.calculation_method == 'percentage' and deduction.rate is not None:
                total_deduction += selling_price * (deduction.rate / Decimal('100.0'))
            elif deduction.calculation_method == 'fixed_amount' and deduction.fixed_amount is not None:
                total_deduction += deduction.fixed_amount

        return selling_price - total_deduction

    def calculate_profit(self, selling_price=None):
        """
        Рассчитать прибыль (чистая выручка минус себестоимость).
        selling_price: Цена, по которой товар был фактически продан.
        """
        net_proceeds = self.calculate_net_proceeds(selling_price)
        cost = self.cost_price or Decimal('0.00')
        return net_proceeds - cost

# --- NEW: Price Type Model ---
class PriceType(BaseModel):
    """Типы цен (розничная, оптовая, себестоимость и т.д.)."""
    name = models.CharField("Название типа цены", max_length=100, unique=True)
    slug = models.SlugField("Slug", max_length=100, unique=True, blank=True)
    description = models.TextField("Описание", blank=True, null=True)
    is_base = models.BooleanField("Базовый тип?", default=False, help_text="Является ли этот тип основной ценой продажи?")
    is_cost = models.BooleanField("Тип себестоимости?", default=False, help_text="Обозначает ли этот тип закупочную цену?")

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Тип цены"
        verbose_name_plural = "Типы цен"


# --- NEW: Price Model ---
class Price(BaseModel):
    """Модель для хранения различных цен на товар."""
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='prices', verbose_name="Товар")
    price_type = models.ForeignKey(PriceType, on_delete=models.PROTECT, related_name='prices', verbose_name="Тип цены")
    value = models.DecimalField("Значение цены", max_digits=12, decimal_places=2) # Увеличил max_digits
    currency = models.CharField("Валюта", max_length=3, default='RUB') # Можно сделать FK на модель Валют
    valid_from = models.DateTimeField("Действует с", null=True, blank=True, db_index=True)
    valid_to = models.DateTimeField("Действует до", null=True, blank=True, db_index=True)
    is_active = models.BooleanField("Активна", default=True, db_index=True, help_text="Является ли эта цена текущей активной для данного типа")

    class Meta:
        verbose_name = "Цена"
        verbose_name_plural = "Цены"
        # Убедимся, что для одного товара и типа цены может быть только одна активная запись без даты окончания
        # Либо используем valid_from/valid_to для управления историей
        indexes = [
            models.Index(fields=['product', 'price_type', 'is_active']),
            models.Index(fields=['valid_from']),
            models.Index(fields=['valid_to']),
        ]
        unique_together = (('product', 'price_type', 'valid_to', 'is_active'),) # Пример ограничения, нужно продумать логику

    def __str__(self):
        return f"{self.product} - {self.price_type}: {self.value} {self.currency}"

# Warehouse Model
class Warehouse(BaseModel):
    name = models.CharField("Название", max_length=255, unique=True)
    name2 = models.CharField("Название 2", max_length=255, blank=True, null=True)
    slug = models.SlugField("Slug (URL)", max_length=255, unique=True, blank=True, help_text="Автоматически генерируется из названия, если пустое")
    description = models.TextField("Описание", blank=True, null=True)
    type = models.CharField("Тип склада", max_length=50, choices=[('physical', 'Физический'), ('virtual', 'Виртуальный')])
    address = models.TextField("Адрес")
    contact_info = models.CharField("Контактная информация", max_length=255, blank=True) # Сделал необязательным
    status = models.CharField("Статус", max_length=20, choices=[('active', 'Активен'), ('inactive', 'Неактивен')], default='active') # default

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Склад"
        verbose_name_plural = "Склады"

# Stock Model
class Stock(BaseModel):
    warehouse = models.ForeignKey(Warehouse, on_delete=models.CASCADE, related_name='stock_items', verbose_name="Склад") # Изменил related_name
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='stock_items', verbose_name="Товар") # Изменил related_name
    quantity = models.DecimalField("Количество", max_digits=10, decimal_places=3, default=0) # Decimal для возможн. дробных единиц
    min_quantity = models.DecimalField("Минимальное количество", max_digits=10, decimal_places=3, default=0)
    last_counted_at = models.DateTimeField("Дата последней инвентаризации", null=True, blank=True)

    class Meta:
        verbose_name = "Остаток на складе"
        verbose_name_plural = "Остатки на складах"
        unique_together = ('warehouse', 'product')
        indexes = [models.Index(fields=['warehouse', 'product'])]

    def __str__(self):
        return f"{self.product.name} - {self.quantity} в {self.warehouse.name}"

# Contragent Model
class Contragent(BaseModel):
    CONTRAGENT_TYPES = (
        ('customer', 'Клиент'),
        ('supplier', 'Поставщик'), # Добавил Поставщика
        ('employee', 'Сотрудник'),
        ('other', 'Другое'), # Добавил Другое
    )
    CONTRAGENT_STATUSES = (
       ('active', 'Активен'),
       ('inactive', 'Неактивен'),
       ('archived', 'В архиве'), # Добавил архив
    )
    # Убрал user как OneToOne, т.к. контрагент не всегда пользователь системы
    # user = models.OneToOneField(User, null=True, blank=True, on_delete=models.SET_NULL, related_name='contragent', verbose_name="Связанный пользователь")
    name = models.CharField("Название/ФИО", max_length=255, unique=True, db_index=True)
    name2 = models.CharField("Название/ФИО 2", max_length=255, blank=True, null=True)
    slug = models.SlugField("Slug", max_length=255, unique=True, blank=True, help_text="Автоматически генерируется из названия, если пустое")
    phone = models.CharField("Телефон", max_length=50, blank=True, null=True)
    email = models.EmailField("Email", blank=True, null=True, db_index=True) # Сделал необязательным и не уникальным глобально
    type = models.CharField("Тип контрагента", max_length=50, choices=CONTRAGENT_TYPES)
    address = models.TextField("Юридический/Фактический адрес", blank=True, null=True)
    status = models.CharField("Статус", max_length=20, choices=CONTRAGENT_STATUSES, default='active')
    # Дополнительные поля для юр.лиц / ИП
    inn = models.CharField("ИНН", max_length=12, blank=True, null=True, db_index=True)
    kpp = models.CharField("КПП", max_length=9, blank=True, null=True)
    # ... другие реквизиты по необходимости

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
            # Добавить проверку уникальности slug перед сохранением
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.name} ({self.get_type_display()})" # Показываем тип в строке

    class Meta:
        verbose_name = "Контрагент"
        verbose_name_plural = "Контрагенты"
        indexes = [
             models.Index(fields=['name']),
             models.Index(fields=['inn']),
             models.Index(fields=['email']),
        ]