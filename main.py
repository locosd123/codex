# -*- coding: utf-8 -*-
import os
import json
import threading
import base64
import shutil
import time
import requests
import random
import sys
import ctypes
import customtkinter as ctk
from tkinter import messagebox, filedialog, simpledialog, Text, END, DISABLED, NORMAL
from urllib.parse import urlparse, urljoin
from PIL import Image
import traceback
# ИСПРАВЛЕНО: Добавлен импорт as_completed и CancelledError
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
import copy
import logging
import queue

# --- Определение базового пути для .exe и .py ---
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.dirname(os.path.realpath(__file__))

# --- Dependencies Check ---
SELENIUM_AVAILABLE = False
TOGETHER_AVAILABLE = False
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException

    SELENIUM_AVAILABLE = True
except ImportError:
    logging.warning("Библиотека Selenium не найдена. Загрузка с URL-страниц может не работать.")

try:
    from together import Together

    TOGETHER_AVAILABLE = True
except ImportError:
    logging.critical("Библиотека 'together' не найдена. Установите: pip install together")

# --- Константы ---
CONFIG_FILE = os.path.join(application_path, "settings.json")
LOG_FILE = os.path.join(application_path, "log.txt")

# Обновленный PROMPT для Llama
PROMPT = (
    "Generate a rich, immersive, and highly detailed description of the image in a bright, cartoony 3D style, "
    "with vivid details and consistent cartoon aesthetic. Pay meticulous attention to anatomical correctness if humans or animals appear: "
    "describe limbs, fingers (e.g., 'a human hand with five fingers'), and posture accurately, ensuring realistic formation "
    "within the stylized context. Guide the generation away from common artifacts like malformed hands or extra limbs. "
    "Transform the original setting into a fictional casino-themed environment — for example, a neon-lit gambling lounge in a cyberpunk city, "
    "a luxury slot machine room on a futuristic space cruiser, or a high-stakes poker suite in a virtual reality metaverse. "
    "Detail every visual element: shapes, volumes, textures, reflections, light sources, shadows, and surface materials. "
    "Use a bold, dynamic color palette distinct from the original scene. "
    "Omit any textual elements found in the image. "
    "The description must be logically consistent, grammatically correct, and fully aligned with the following: "
    "styled as bright, cartoony 3D; transformed to a casino theme; anatomically accurate; artifact-free; bold colors; deep visual detail."
)

# Обновленные DEFAULT_PREFIXES
DEFAULT_PREFIXES = [
    "Masterpiece, best quality, ultra-detailed, intricate details, sharp focus, high resolution, bright-cartoon 3D style. "
    "Maintain the same main character while varying their position and location. Emphasize the casino theme and remove unnecessary logos or text. Image:",
    "Masterpiece, best quality, ultra-detailed, intricate details, sharp focus, high resolution, bright-cartoon 3D style. "
    "Ensure the background is unique, the main character remains consistent, and the scene emphasizes casino elements. Remove all extraneous text and logos. Image:",
    "Masterpiece, best quality, ultra-detailed, intricate details, sharp focus, high resolution, bright-cartoon 3D style. "
    "Keep the main character identical but change their location and pose for each prompt. Highlight the casino setting and remove irrelevant logos or inscriptions. Image:"
]
DEFAULT_PREFIXES_LEN = len(DEFAULT_PREFIXES)

# --- НОВЫЕ КОНСТАНТЫ для FLUX.1-schnell ---
FLUX_MODEL_NAME = "black-forest-labs/FLUX.1-schnell"
FLUX_IMAGE_WIDTH  = 1024
FLUX_IMAGE_HEIGHT = 576
FLUX_IMAGE_STEPS = 12  # Уменьшено до 12 для баланса скорости и качества (можно вернуть 20 если нужно)
FLUX_IMAGE_N = 1
SEED_NUMBER = 1

# --- Константы для Leonardo AI ---
LEONARDO_API_URL = "https://cloud.leonardo.ai/api/rest/v1"
LEONARDO_IMAGE_WIDTH = 1024
LEONARDO_IMAGE_HEIGHT = 576
LEONARDO_IMAGE_STEPS = 20
LEONARDO_POLL_DELAY = 2.0
LEONARDO_MAX_POLLS = 30

DEFAULT_NEGATIVE_PROMPT = (
    "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, "
    "blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, low quality, jpeg artifacts, noisy, weird colors, "
    "malformed limbs, extra fingers, mutated hands, too many fingers, fused fingers, missing limbs, distorted hands, "
    "text, letters, words, captions, logos, inscriptions, writing, lowres, bad proportions, "
    "bad composition, extra_legs, extra_arms, more_than_two_hands_per_person, more_than_two_legs_per_person, "
    "more_than_five_fingers_on_one_hand, fewer_than_five_fingers_on_one_hand, fused_limbs, crooked_limbs, "
    "mangled_limbs, cloned_face, error, duplicate, morbid, mutilated, "
    "gross, disgusting, poorly detailed, simple background, boring background, "
    "missing_fingers, bad_hands, extra_digits, fused_digits, poorly_lit, overexposed, underexposed, "
    "mismatched_eyes, asymmetrical_face, conjoined, nsfw, nude, explicit"
)

# --- Параметры контроля нагрузки для FLUX ---
FLUX_MAX_CONCURRENT = 3
FLUX_BASE_DELAY = 1.0
FLUX_MAX_RETRIES = 4
flux_semaphore = threading.Semaphore(FLUX_MAX_CONCURRENT)

script_dir = application_path
images_dir = os.path.join(script_dir, "Images")
os.makedirs(images_dir, exist_ok=True)

from webdriver_manager.chrome import ChromeDriverManager

driver_name = 'chromedriver.exe' if os.name == 'nt' else 'chromedriver'
driver_path = os.path.join(script_dir, driver_name)


def get_chromedriver_path() -> str:
    # Исправлено: добавлен явный путь для менеджера драйверов, чтобы избежать потенциальных проблем с доступом
    try:
        driver_install_path = os.path.join(application_path, "chromedriver_cache")
        os.makedirs(driver_install_path, exist_ok=True)
        downloaded = ChromeDriverManager(path=driver_install_path).install()
        if os.path.abspath(downloaded) != os.path.abspath(driver_path):
            try:
                shutil.copy(downloaded, driver_path)
                log_message(f"ChromeDriver скопирован в: {driver_path}", level=logging.DEBUG)
            except Exception as copy_err:
                log_message(
                    f"Не удалось скопировать ChromeDriver в {driver_path}: {copy_err}. Используется путь: {downloaded}",
                    level=logging.WARNING)
                return downloaded  # Возвращаем путь из кэша менеджера
        return driver_path
    except Exception as e:
        log_message(f"Ошибка при установке/копировании ChromeDriver: {e}. Попытка использовать системный chromedriver.",
                    level=logging.ERROR)
        # В случае полной неудачи, вернем просто имя файла, полагаясь на системный PATH
        return driver_name


# --- Настройка логирования ---
try:
    with open(LOG_FILE, "w", encoding='utf-8') as f:
        f.write(f"--- Log started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
except IOError as e:
    print(f"ERROR: Could not clear log file {LOG_FILE}: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- Глобальные переменные для GUI и управления папками ---
app = None
folder_rows_frame = None
folder_data = []
log_textbox = None
log_queue = queue.Queue()

stop_requested = threading.Event()


# --- Вспомогательные функции ---
def log_message(message: str, level=logging.INFO):
    try:
        if level == logging.CRITICAL:
            logging.critical(message)
        elif level == logging.ERROR:
            logging.error(message)
        elif level == logging.WARNING:
            logging.warning(message)
        else:
            logging.info(message)
        log_queue.put(message)
    except Exception as e:
        print(f"Logging error: {e}\nOriginal message: {message}")


def update_log_textbox():
    global log_textbox, app
    try:
        while not log_queue.empty():
            message = log_queue.get_nowait()
            if log_textbox and log_textbox.winfo_exists():
                log_textbox.configure(state=NORMAL)
                log_textbox.insert(END, message + "\n")
                log_textbox.see(END)
                log_textbox.configure(state=DISABLED)
    except queue.Empty:
        pass
    except Exception as e:
        print(f"Error updating log textbox: {e}")
    finally:
        if app: app.after(100, update_log_textbox)


def load_settings():
    global folder_data
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding="utf-8") as f:
                settings = json.load(f)
                loaded_folders = settings.get("folders", [])
                # Исправлено: Загрузка данных без информации о виджетах
                folder_data = [{'name': f_data.get('name', 'Unnamed'),
                                'url': f_data.get('url', ''),
                                'names': f_data.get('names', ''),
                                'quantity': f_data.get('quantity', 1),
                                'checked': f_data.get('checked', True)}
                               for f_data in loaded_folders if f_data.get('name')]
                log_message(f"Загружено {len(folder_data)} папок из настроек.")
                return settings
        except (json.JSONDecodeError, IOError) as e:
            log_message(f"Ошибка загрузки настроек ({CONFIG_FILE}): {e}", level=logging.ERROR)
            folder_data = []  # Очищаем данные при ошибке
    else:
        log_message("Файл настроек не найден. Будут использованы значения по умолчанию.")
        folder_data = []  # Убеждаемся, что папки пусты, если файла нет
    return {}


def save_settings():
    global folder_data, app
    if not app or not entry_together.winfo_exists():
        log_message("Попытка сохранить настройки до полной инициализации GUI. Пропуск.", level=logging.WARNING)
        return {}

    current_folder_config = []
    for row_data in folder_data:  # Итерируемся по списку словарей folder_data
        # Проверяем наличие ключей GUI элементов перед их чтением
        if row_data.get('frame') and row_data['frame'].winfo_exists() \
                and row_data.get('label') and row_data['label'].winfo_exists() \
                and row_data.get('url_entry') and row_data['url_entry'].winfo_exists() \
                and row_data.get('names_entry') and row_data['names_entry'].winfo_exists() \
                and row_data.get('quantity_entry') and row_data['quantity_entry'].winfo_exists() \
                and row_data.get('checkbox') and row_data['checkbox'].winfo_exists():

            quantity_str = row_data['quantity_entry'].get().strip()
            try:
                quantity = int(quantity_str);
                quantity = max(1, quantity)
            except ValueError:
                quantity = 1

            # Сохраняем только данные, без ссылок на GUI виджеты
            current_folder_config.append({
                'name': row_data['label'].cget("text"),  # Берем текст из метки
                'url': row_data['url_entry'].get().strip(),
                'names': row_data['names_entry'].get().strip(),
                'quantity': quantity,
                'checked': bool(row_data['checkbox'].get())
            })
        else:
            # Если виджеты не найдены (например, папка была удалена),
            # сохраняем только данные, которые у нас есть в словаре row_data,
            # исключая ссылки на виджеты.
            folder_data_to_save = {k: v for k, v in row_data.items() if k not in [
                'url_entry', 'names_entry', 'quantity_entry', 'checkbox',
                'frame', 'label', 'edit_button', 'delete_button'
            ]}
            # Проверяем, есть ли обязательные поля
            if folder_data_to_save.get('name'):
                # Убедимся, что quantity и checked имеют значения по умолчанию, если их нет
                folder_data_to_save.setdefault('quantity', 1)
                folder_data_to_save.setdefault('checked', True)
                current_folder_config.append(folder_data_to_save)
                log_message(
                    f"Сохранение данных для папки '{folder_data_to_save.get('name')}' без информации о GUI (возможно, удалена).",
                    level=logging.DEBUG)

    threads_value = "1"
    if entry_threads and entry_threads.winfo_exists():
        threads_value = entry_threads.get().strip()

    cfg = {
        "together_api": entry_together.get().strip(),
        "leonardo_api": entry_leonardo.get().strip(),
        "save_dir": entry_save_dir.get().strip(),
        "threads": threads_value,
        "default_image_url": entry_url_default.get().strip(),
        "default_names": entry_names_default.get().strip(),
        "folders": current_folder_config  # Сохраняем очищенный список папок
    }
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)
        log_message("Настройки сохранены.", level=logging.DEBUG)
    except IOError as e:
        messagebox.showerror("Ошибка Сохранения", f"Не удалось сохранить настройки:\n{e}")
        log_message(f"Ошибка сохранения настроек: {e}", level=logging.ERROR)
    return cfg


def is_valid_url(url: str) -> bool:
    try:
        p = urlparse(url);
        return p.scheme in ("http", "https") and bool(p.netloc)
    except ValueError:
        return False


def browse_dir():
    d = filedialog.askdirectory(title="Выберите папку для сохранения результатов")
    if d: entry_save_dir.delete(0, ctk.END); entry_save_dir.insert(0, d)


def browse_file_default():
    path = filedialog.askopenfilename(title="Выберите файл изображения",
                                      filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.gif"),
                                                 ("All files", "*.*")])
    if path: entry_url_default.delete(0, ctk.END); entry_url_default.insert(0, path)


def browse_file_folder(url_entry_widget):
    path = filedialog.askopenfilename(title="Выберите файл изображения для папки",
                                      filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.gif"),
                                                 ("All files", "*.*")])
    if path: url_entry_widget.delete(0, ctk.END); url_entry_widget.insert(0, path)


def cleanup_temp_files(*files):
    for file_path in files:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path);
                log_message(f"Временный файл удален: {os.path.basename(file_path)}",
                            level=logging.DEBUG)
            except OSError as e:
                log_message(f"Не удалось удалить временный файл {os.path.basename(file_path)}: {e}",
                            level=logging.WARNING)


# --- Функция для API FLUX.1-schnell ---
def call_flux_schnell_with_retry(prompt: str, together_key: str):
    if stop_requested.is_set():
        raise RuntimeError("Генерация FLUX.1-schnell прервана пользователем перед запросом.")

    last_exception = None
    for attempt in range(FLUX_MAX_RETRIES):
        if stop_requested.is_set():
            raise RuntimeError(
                f"Генерация FLUX.1-schnell прервана пользователем во время ожидания (попытка {attempt + 1}).")

        acquired = flux_semaphore.acquire(timeout=60)
        if not acquired:
            last_exception = TimeoutError("Timeout waiting for FLUX.1-schnell semaphore")
            log_message(f"FLUX.1-schnell: Таймаут ожидания семафора (попытка {attempt + 1})", level=logging.WARNING)
            continue

        try:
            delay = FLUX_BASE_DELAY * (2 ** attempt)
            log_message(f"FLUX.1-schnell: Попытка {attempt + 1}/{FLUX_MAX_RETRIES}, задержка {delay:.2f} сек...")

            time.sleep(min(delay, 1.0))
            if stop_requested.is_set():
                raise RuntimeError(
                    f"Генерация FLUX.1-schnell прервана пользователем перед запросом (попытка {attempt + 1}).")

            client = Together(api_key=together_key)
            response = client.images.generate(
                prompt=prompt,
                model=FLUX_MODEL_NAME,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,  # Добавлено
                width=FLUX_IMAGE_WIDTH,
                height=FLUX_IMAGE_HEIGHT,
                steps=FLUX_IMAGE_STEPS,  # Используется обновленное значение
                n=FLUX_IMAGE_N,
                seed=SEED_NUMBER,
                response_format="b64_json",
            )

            if response and response.data and len(response.data) > 0 and response.data[0].b64_json:
                log_message(f"FLUX.1-schnell: Изображение успешно сгенерировано (попытка {attempt + 1}).")
                return response.data[0].b64_json
            else:
                # Попытка получить детали ошибки из ответа, если они есть
                error_detail = getattr(response, 'error', str(response))
                log_message(f"FLUX.1-schnell: Некорректный ответ от API (попытка {attempt + 1}): {error_detail}",
                            level=logging.ERROR)
                last_exception = RuntimeError(f"FLUX.1-schnell: API returned unexpected response: {error_detail}")
                # Можно добавить break, если ответ явно указывает на неповторяемую ошибку

        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code if http_err.response else None
            try:
                # Попытка извлечь сообщение об ошибке из JSON тела ответа
                error_details = http_err.response.json()
                error_text = error_details.get('message', str(http_err))  # Часто ошибка в поле 'message'
                error_type = error_details.get('type_', '')  # Together иногда использует 'type_'
            except json.JSONDecodeError:
                error_text = http_err.response.text if http_err.response else str(http_err)
                error_type = ''

            log_message(
                f"FLUX.1-schnell: HTTP ошибка {status_code}: {error_text} (Type: {error_type}, Попытка {attempt + 1}).",
                level=logging.ERROR)
            last_exception = http_err

            # --- ИСПРАВЛЕНО: Обработка NSFW ошибки ---
            if status_code == 422 and "nsfw content" in error_text.lower():
                log_message(f"FLUX.1-schnell: Ошибка 422 (NSFW): {error_text}. Промпт: {prompt[:100]}...",
                            level=logging.WARNING)
                break  # Прерываем ретраи для этой ошибки
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
            elif status_code == 400:  # Bad Request (например, невалидный параметр)
                log_message(f"FLUX.1-schnell: Ошибка 400 Bad Request: {error_text}. Промпт: {prompt[:100]}...",
                            level=logging.ERROR)
                break  # Не повторяем при 400 (не Rate Limit)
            elif status_code == 429 or (status_code and status_code >= 500):  # Rate Limit или Серверная ошибка
                log_message(f"FLUX.1-schnell: Ошибка {status_code}, повтор...", level=logging.WARNING)
                # Продолжаем цикл (неявный continue)
            else:  # Другие HTTP ошибки
                break  # Не повторяем
        except requests.exceptions.RequestException as req_err:
            log_message(f"FLUX.1-schnell: Ошибка сети или запроса (попытка {attempt + 1}): {req_err}",
                        level=logging.ERROR)
            last_exception = req_err
        except Exception as e:  # Другие непредвиденные ошибки (например, от библиотеки Together)
            log_message(f"FLUX.1-schnell: Непредвиденная ошибка (попытка {attempt + 1}): {type(e).__name__} - {e}",
                        level=logging.CRITICAL)
            logging.exception("Traceback непредвиденной ошибки FLUX:")
            last_exception = e
            break  # Не повторяем при неизвестных ошибках
        finally:
            if acquired:  # Освобождаем семафор только если он был захвачен
                flux_semaphore.release()
                log_message(f"FLUX.1-schnell: Семафор освобожден (попытка {attempt + 1})", level=logging.DEBUG)

    # Выход из цикла ретраев
    error_msg = "FLUX.1-schnell: Не удалось сгенерировать изображение после всех попыток."
    if last_exception:
        # Передаем последнюю ошибку для более детального логгирования в worker
        raise RuntimeError(
            f"{error_msg} Последняя ошибка: {type(last_exception).__name__} - {last_exception}") from last_exception
    else:
        # Если вышли из цикла без исключений, но и без результата (маловероятно, но возможно)
        raise RuntimeError(error_msg)


# --- Функция для генерации изображений через Leonardo AI ---
def call_leonardo_with_retry(prompt: str, leonardo_key: str):
    if stop_requested.is_set():
        raise RuntimeError("Генерация Leonardo AI прервана пользователем перед запросом.")

    headers = {
        "Authorization": f"Bearer {leonardo_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "width": LEONARDO_IMAGE_WIDTH,
        "height": LEONARDO_IMAGE_HEIGHT,
        "num_images": 1,
        "num_inference_steps": LEONARDO_IMAGE_STEPS,
    }

    try:
        resp = requests.post(f"{LEONARDO_API_URL}/generations", headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        job = resp.json().get("sdGenerationJob", {})
        gen_id = job.get("generationId")
        if not gen_id:
            raise RuntimeError("Leonardo API не вернул идентификатор генерации.")

        poll_url = f"{LEONARDO_API_URL}/generations/{gen_id}"
        for _ in range(LEONARDO_MAX_POLLS):
            if stop_requested.is_set():
                raise RuntimeError("Генерация Leonardo AI прервана пользователем во время ожидания.")
            poll_resp = requests.get(poll_url, headers=headers, timeout=60)
            poll_resp.raise_for_status()
            data = poll_resp.json()
            status = data.get("generations_by_pk", {}).get("status")
            if status == "COMPLETE":
                images = data.get("generations_by_pk", {}).get("generated_images", [])
                if images:
                    img_url = images[0].get("url")
                    if img_url:
                        img_resp = requests.get(img_url, timeout=60)
                        img_resp.raise_for_status()
                        return base64.b64encode(img_resp.content).decode()
                break
            time.sleep(LEONARDO_POLL_DELAY)
        raise RuntimeError("Leonardo API: превышено время ожидания результата.")
    except Exception as e:
        raise RuntimeError(f"Leonardo API error: {e}") from e

# --- Функция рабочего потока ---
def worker(task_id, task_data):
    image_path_or_url = task_data['image_path_or_url']
    together_key = task_data.get('together_key')
    leonardo_key = task_data.get('leonardo_key')
    target_save_dir = task_data['save_dir']
    names_list = task_data['names_list']
    is_folder_task = task_data.get('is_folder_task', False)
    folder_name = task_data.get('folder_name', '')

    log_prefix = f"Задача {task_id}" + (f" (Папка: {folder_name})" if is_folder_task else "") + ":"

    max_retries = 3  # Количество попыток для ВСЕЙ задачи worker (включая скачивание, обработку, API вызовы)
    original_image_local_path = None
    converted_jpg_path = None
    data_uri = None

    for attempt in range(1, max_retries + 1):
        try:
            if stop_requested.is_set():
                raise RuntimeError("Задача прервана пользователем перед началом.")

            use_together = bool(together_key)
            use_leonardo = bool(leonardo_key)

            if not use_together and not use_leonardo:
                raise ValueError(f"{log_prefix} Не указан API ключ для генерации.")
            if use_together and use_leonardo:
                raise ValueError(f"{log_prefix} Укажите только один API ключ (Leonardo или Together).")

            if use_together and not TOGETHER_AVAILABLE:
                raise ImportError(f"{log_prefix} Библиотека 'together' не установлена.")

            # Шаг 1: Получение исходного изображения
            log_message(f"{log_prefix} Загрузка изображения из: {image_path_or_url} (Попытка worker {attempt})")
            original_image_local_path = fetch_image(image_path_or_url)

            if stop_requested.is_set():
                raise RuntimeError("Задача прервана пользователем после загрузки изображения.")

            # Шаг 2: Конвертация и создание Data URI
            img = Image.open(original_image_local_path)
            if img.mode == 'RGBA':
                log_message(f"{log_prefix} Конвертация RGBA в RGB...")
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg
            elif img.mode != 'RGB':
                log_message(f"{log_prefix} Конвертация из {img.mode} в RGB...")
                img = img.convert('RGB')

            base_name_orig = os.path.splitext(os.path.basename(original_image_local_path))[0]
            # Добавляем попытку в имя временного файла для уникальности при ретраях worker
            unique_suffix = f"_{task_id}_w{attempt}_{int(time.time())}"
            converted_jpg_path = os.path.join(images_dir, f"{base_name_orig}_converted{unique_suffix}.jpg")
            img.save(converted_jpg_path, "JPEG", quality=90)
            log_message(f"{log_prefix} Изображение конвертировано в JPEG: {os.path.basename(converted_jpg_path)}",
                        level=logging.DEBUG)

            with open(converted_jpg_path, "rb") as f:
                b64_string = base64.b64encode(f.read()).decode('utf-8')
            data_uri = f"data:image/jpeg;base64,{b64_string}"
            log_message(f"{log_prefix} Data URI создан (длина: {len(data_uri)}).", level=logging.DEBUG)

            if use_together:
                if stop_requested.is_set():
                    raise RuntimeError("Задача прервана пользователем перед запросом к Together.ai Chat.")

                # Шаг 3: Получение улучшенного промпта через Together Chat
                log_message(f"{log_prefix} Запрос к Together.ai Chat для улучшения промпта...")
                chat_client = Together(api_key=together_key)
                try:
                    resp_chat = chat_client.chat.completions.create(
                        model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                        messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT},
                                                               {"type": "image_url", "image_url": {"url": data_uri}}]}],
                        stream=False,
                        max_tokens=1024
                    )
                    if not resp_chat.choices:
                        raise ValueError("Chat API вернул ответ без 'choices'.")
                    together_prompt_content = resp_chat.choices[0].message.content.strip()
                    if not together_prompt_content:
                        raise ValueError("Chat API вернул пустое сообщение в 'choices'.")
                    log_message(f"{log_prefix} Получен промпт от Together.ai Chat (длина {len(together_prompt_content)}).")
                except Exception as chat_err:
                    log_message(f"{log_prefix} Ошибка при запросе к Together.ai Chat: {chat_err}", level=logging.ERROR)
                    raise RuntimeError(
                        f"Ошибка получения промпта от Chat API: {chat_err}") from chat_err

                # Шаг 4: Формирование финального промпта
                selected_prefix = random.choice(DEFAULT_PREFIXES)
                final_prompt_for_image = selected_prefix + together_prompt_content
                log_message(
                    f"{log_prefix} Финальный промпт для FLUX.1-schnell (длина {len(final_prompt_for_image)}): {final_prompt_for_image[:200]}...",
                    level=logging.DEBUG)

                if stop_requested.is_set():
                    raise RuntimeError("Задача прервана пользователем перед запросом к FLUX.1-schnell.")

                log_message(f"{log_prefix} Отправка промпта в FLUX.1-schnell...")
                b64_image_data = call_flux_schnell_with_retry(final_prompt_for_image, together_key)
            else:
                # Используем Leonardo AI для генерации
                final_prompt_for_image = PROMPT
                log_message(f"{log_prefix} Отправка промпта в Leonardo AI...")
                b64_image_data = call_leonardo_with_retry(final_prompt_for_image, leonardo_key)

            if not b64_image_data:  # Дополнительная проверка, хотя call_flux... должен вызвать исключение
                raise RuntimeError(f"{log_prefix} FLUX.1-schnell не вернул данные изображения после ретраев API.")
            log_message(f"{log_prefix} Сгенерировано изображение (base64, длина: {len(b64_image_data)}).")

            if stop_requested.is_set():
                raise RuntimeError("Задача прервана пользователем перед сохранением результата.")

            # Шаг 6: Сохранение результата
            image_bytes = base64.b64decode(b64_image_data)

            base_name_save = names_list[0] if names_list else f"generated_image_{task_id}"
            safe_base = "".join(c for c in base_name_save if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ",
                                                                                                                  "_")
            if not safe_base: safe_base = f"image_{task_id}"
            ext = '.png'

            os.makedirs(target_save_dir, exist_ok=True)

            # Исправление пути сохранения для Windows
            # target_save_dir должен быть абсолютным путем или корректно обрабатываться os.path.join
            # Проверим, что main_save_dir - корректный путь
            if not os.path.isdir(target_save_dir):
                log_message(
                    f"{log_prefix} Ошибка: Целевая директория '{target_save_dir}' не найдена или не является директорией.",
                    level=logging.ERROR)
                # Можно попробовать создать ее, но лучше убедиться, что исходный путь корректен
                try:
                    os.makedirs(target_save_dir, exist_ok=True)
                    log_message(f"{log_prefix} Целевая директория '{target_save_dir}' создана.", level=logging.INFO)
                except Exception as mkdir_err:
                    raise IOError(
                        f"Не удалось создать целевую директорию '{target_save_dir}': {mkdir_err}") from mkdir_err

            out_path = os.path.join(target_save_dir, f"{safe_base}{ext}")
            counter = 1
            # Чтобы избежать перезаписи при ретраях worker, добавим суффикс попытки worker если нужно
            original_out_path = out_path
            while os.path.exists(out_path):
                # Если файл существует, добавляем суффикс _<counter>
                suffix = f"_{counter}"
                out_path = os.path.join(target_save_dir, f"{safe_base}{suffix}{ext}")
                counter += 1
                if counter > 100 + len(names_list):  # Ограничение на количество копий
                    raise RuntimeError(
                        f"{log_prefix} Слишком много файлов с похожими именами '{safe_base}' в папке: {target_save_dir}")

            log_message(f"{log_prefix} Сохранение в: {out_path}", level=logging.DEBUG)
            with open(out_path, "wb") as f:
                f.write(image_bytes)

            log_message(f"{log_prefix} Успешно сохранено: {out_path}")
            # Если все успешно, выходим из цикла ретраев worker
            return {"success": True, "path": out_path, "task_id": task_id}

        except Exception as e:
            import errno
            if isinstance(e, OSError) and getattr(e, "winerror", None) == 123:  # Ошибка имени файла
                log_message(f"{log_prefix} Недопустимое имя файла: {e}", level=logging.ERROR)
                # Не повторяем попытку worker для этой ошибки
                return {"success": False, "error": str(e), "task_id": task_id}

            cancelled = isinstance(e, RuntimeError) and "прервано пользователем" in str(e).lower()
            if cancelled:
                log_message(f"{task_id}: прервано пользователем на попытке {attempt} задачи worker",
                            level=logging.WARNING)
                # Не повторяем попытку worker
                return {"success": False, "error": str(e), "task_id": task_id, "cancelled": True}

            # Логируем ошибку текущей попытки worker
            log_message(f"{task_id}: ошибка на попытке {attempt}/{max_retries} задачи worker: {type(e).__name__} - {e}",
                        level=logging.ERROR)
            # Логируем traceback только для действительно неожиданных ошибок
            if not isinstance(e, (
            requests.exceptions.RequestException, TimeoutError, RuntimeError, IOError, ImportError, ValueError)):
                logging.exception(f"Traceback для задачи {task_id} на попытке {attempt}:")

            if attempt == max_retries:
                # Если это была последняя попытка worker, логируем это и пробрасываем исключение
                log_message(f"{task_id}: Все {max_retries} попыток задачи worker не удались.", level=logging.CRITICAL)
                raise  # Это приведет к ошибке в future.result() в generate_thread

            # Задержка перед следующей попыткой worker
            # Увеличиваем задержку с каждой попыткой
            worker_retry_delay = 1.5 * attempt
            log_message(f"{task_id}: Ожидание {worker_retry_delay:.1f} сек перед следующей попыткой worker...",
                        level=logging.DEBUG)
            time.sleep(worker_retry_delay)

        finally:
            # Очистка временных файлов после каждой попытки worker
            cleanup_temp_files(original_image_local_path, converted_jpg_path)
            original_image_local_path = None  # Сбрасываем пути
            converted_jpg_path = None

    # Этот код не должен достигаться, если max_retries > 0,
    # так как последняя неудачная попытка вызовет raise
    return {"success": False,
            "error": f"Не удалось обработать задачу {task_id} после {max_retries} попыток (неожиданный выход из цикла worker).",
            "task_id": task_id}


# --- fetch_image (без изменений) ---
def fetch_image(path_or_url: str) -> str:
    log_message(f"Загрузка изображения из: {path_or_url}", level=logging.DEBUG)
    temp_image_path = None

    if stop_requested.is_set(): raise RuntimeError("Загрузка изображения прервана пользователем.")

    if os.path.isfile(path_or_url):
        try:
            base_filename = os.path.basename(path_or_url)
            safe_base = "".join(c for c in os.path.splitext(base_filename)[0] if c.isalnum() or c in ('_', '-'))[:50]
            _, ext = os.path.splitext(base_filename)
            unique_suffix = f"_{threading.get_ident()}_{time.time():.0f}"
            temp_image_path = os.path.join(images_dir, f"input_{safe_base}{unique_suffix}{ext if ext else '.tmp'}")
            # Добавлена проверка существования директории перед копированием
            os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
            shutil.copy(path_or_url, temp_image_path)
            log_message(f"Локальный файл скопирован: {temp_image_path}")
            return temp_image_path
        except Exception as e:
            raise IOError(f"Не удалось скопировать локальный файл '{path_or_url}': {e}") from e

    elif is_valid_url(path_or_url):
        try:
            if stop_requested.is_set(): raise RuntimeError("Прямое скачивание URL прервано пользователем.")
            response = requests.get(path_or_url, stream=True, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            content_type = response.headers.get('content-type', '').lower()
            if 'image' not in content_type: raise ValueError(f"Content-Type ({content_type}) не является изображением.")
            parsed_url = urlparse(path_or_url);
            filename = os.path.basename(parsed_url.path) or "download";
            _, ext = os.path.splitext(filename)
            known_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
            if not ext or ext.lower() not in known_extensions:
                mime_to_ext = {'image/jpeg': '.jpg', 'image/png': '.png', 'image/webp': '.webp', 'image/gif': '.gif',
                               'image/bmp': '.bmp'}
                ext = mime_to_ext.get(content_type, '.png')

            safe_filename = "".join(c for c in os.path.splitext(filename)[0] if c.isalnum() or c in ('_', '-'))[:50]
            unique_suffix = f"_{threading.get_ident()}_{time.time():.0f}"
            temp_image_path = os.path.join(images_dir, f"download_{safe_filename}{unique_suffix}{ext}")
            # Добавлена проверка существования директории
            os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
            with open(temp_image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if stop_requested.is_set():
                        cleanup_temp_files(temp_image_path)
                        raise RuntimeError("Прямое скачивание URL прервано пользователем во время записи.")
                    f.write(chunk)
            log_message(f"URL успешно скачан (прямой запрос): {temp_image_path}")
            return temp_image_path
        except (requests.exceptions.RequestException, ValueError, RuntimeError) as e:
            log_message(f"Прямое скачивание URL не удалось ({e}). Пробуем Selenium (если доступен)...",
                        level=logging.WARNING)
            cleanup_temp_files(temp_image_path);
            temp_image_path = None
            if isinstance(e, RuntimeError) and "прервано пользователем" in str(e).lower():
                raise e

        if temp_image_path is None:
            if not SELENIUM_AVAILABLE: raise ImportError(
                f"Прямое скачивание не удалось, а Selenium недоступен для URL: {path_or_url}")
            log_message("Используем Selenium для загрузки изображения...")
            driver = None
            try:
                if stop_requested.is_set(): raise RuntimeError("Запуск Selenium прерван пользователем.")
                opts = Options();
                opts.add_argument("--headless");
                opts.add_argument("--disable-gpu");
                opts.add_argument("--window-size=1280,1024")
                opts.add_argument(
                    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36")
                opts.add_argument("--no-sandbox");
                opts.add_argument("--disable-dev-shm-usage");
                opts.add_argument("--log-level=3")
                # Добавим опцию для игнорирования ошибок сертификатов, если нужно
                opts.add_argument('--ignore-certificate-errors')
                opts.add_argument('--allow-running-insecure-content')

                try:
                    service = ChromeService(executable_path=get_chromedriver_path())
                    driver = webdriver.Chrome(service=service, options=opts)
                    # Установим таймаут загрузки страницы
                    driver.set_page_load_timeout(45)
                except WebDriverException as wd_err:
                    log_message(f"Ошибка инициализации WebDriver: {wd_err}", level=logging.ERROR)
                    if "executable needs to be in PATH" in str(wd_err):
                        raise WebDriverException(
                            f"ChromeDriver не найден в PATH.\nУстановите ChromeDriver или укажите путь к нему.\nОшибка: {wd_err}") from wd_err
                    else:
                        raise  # Перебрасываем другие ошибки WebDriver

                try:
                    driver.get(path_or_url)
                except TimeoutException as page_load_timeout:
                    # Если страница не загрузилась за 45 секунд, попробуем все равно найти img
                    log_message(f"Selenium: Таймаут загрузки страницы {path_or_url}, но попробуем найти <img>.",
                                level=logging.WARNING)
                except WebDriverException as get_err:
                    # Обработка других ошибок при driver.get()
                    raise RuntimeError(f"Selenium: Ошибка при загрузке URL '{path_or_url}': {get_err}") from get_err

                wait = WebDriverWait(driver, 20)  # Таймаут ожидания элемента
                log_message(f"Selenium: Страница {path_or_url} загружена (или таймаут), ищем <img>...",
                            level=logging.DEBUG)

                if stop_requested.is_set(): raise RuntimeError("Ожидание элемента Selenium прервано пользователем.")
                img_elem = wait.until(EC.visibility_of_element_located((By.TAG_NAME, "img")))
                log_message("Selenium: Найден видимый элемент <img>.")
                img_src = img_elem.get_attribute('src');
                downloaded_from_src = False
                if img_src:
                    log_message(f"Selenium: Найден src: {img_src[:100]}...", level=logging.DEBUG)
                    img_src_abs = urljoin(path_or_url, img_src);
                    log_message(f"Selenium: Абсолютный src URL: {img_src_abs}", level=logging.DEBUG)
                    if is_valid_url(img_src_abs):
                        try:
                            if stop_requested.is_set(): raise RuntimeError(
                                "Скачивание из src Selenium прервано пользователем.")
                            src_resp = requests.get(img_src_abs, stream=True, timeout=30,
                                                    headers={'User-Agent': 'Mozilla/5.0'});
                            src_resp.raise_for_status()
                            content_type = src_resp.headers.get('content-type', '').lower()
                            if 'image' not in content_type: raise ValueError(
                                f"URL из src не вернул изображение (Content-Type: {content_type})")
                            _, src_ext = os.path.splitext(urlparse(img_src_abs).path)
                            if not src_ext or src_ext.lower() not in known_extensions:
                                mime_to_ext = {'image/jpeg': '.jpg', 'image/png': '.png', 'image/webp': '.webp',
                                               'image/gif': '.gif', 'image/bmp': '.bmp'}
                                src_ext = mime_to_ext.get(content_type, '.png')
                            unique_suffix = f"_{threading.get_ident()}_{time.time():.0f}"
                            temp_image_path = os.path.join(images_dir, f"selenium_src_download{unique_suffix}{src_ext}")
                            # Добавлена проверка существования директории
                            os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
                            with open(temp_image_path, 'wb') as f:
                                for chunk in src_resp.iter_content(chunk_size=8192):
                                    if stop_requested.is_set():
                                        cleanup_temp_files(temp_image_path)
                                        raise RuntimeError(
                                            "Скачивание из src Selenium прервано пользователем во время записи.")
                                    f.write(chunk)
                            log_message(f"Selenium: Изображение успешно скачано из src: {temp_image_path}")
                            downloaded_from_src = True
                        except (requests.exceptions.RequestException, ValueError, IOError, RuntimeError) as src_err:
                            log_message(
                                f"Selenium: Не удалось скачать из src '{img_src_abs}': {src_err}. Пробуем скриншот...",
                                level=logging.WARNING)
                            cleanup_temp_files(temp_image_path);
                            temp_image_path = None
                            if isinstance(src_err, RuntimeError) and "прервано пользователем" in str(src_err).lower():
                                raise src_err
                    else:
                        log_message(f"Selenium: Сконструированный URL из src невалиден: {img_src_abs}",
                                    level=logging.WARNING)

                if not downloaded_from_src:
                    log_message("Selenium: Делаем скриншот элемента <img>...")
                    if stop_requested.is_set(): raise RuntimeError(
                        "Создание скриншота Selenium прервано пользователем.")
                    try:
                        # Попытка прокрутить к элементу перед скриншотом
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});",
                                              img_elem)
                        time.sleep(0.5)  # Пауза после скролла
                    except Exception as scroll_err:
                        log_message(f"Selenium: Не удалось прокрутить элемент (не критично): {scroll_err}",
                                    level=logging.WARNING)
                    unique_suffix = f"_{threading.get_ident()}_{time.time():.0f}"
                    temp_image_path = os.path.join(images_dir, f"selenium_screenshot{unique_suffix}.png")
                    try:
                        png_data = img_elem.screenshot_as_png
                        if not png_data: raise RuntimeError("screenshot_as_png вернул пустые данные.")
                        # Добавлена проверка существования директории
                        os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
                        with open(temp_image_path, 'wb') as f:
                            f.write(png_data)
                        if not os.path.exists(temp_image_path) or os.path.getsize(temp_image_path) == 0:
                            cleanup_temp_files(temp_image_path)
                            raise RuntimeError("Файл скриншота не создан или пуст после записи.")
                        log_message(f"Selenium: Скриншот элемента сохранен: {temp_image_path}")
                    except Exception as screenshot_err:
                        cleanup_temp_files(temp_image_path)
                        raise RuntimeError(
                            f"Не удалось сделать скриншот элемента: {screenshot_err}") from screenshot_err

                if temp_image_path and os.path.exists(temp_image_path) and os.path.getsize(temp_image_path) > 0:
                    return temp_image_path
                else:
                    cleanup_temp_files(temp_image_path)
                    raise RuntimeError("Не удалось получить изображение через Selenium.")
            # Обрабатываем специфичные ошибки Selenium
            except TimeoutException as e:
                raise RuntimeError(
                    f"Selenium: Не удалось найти/дождаться элемента <img> на '{path_or_url}' за {wait.timeout} сек. {e}") from e
            except WebDriverException as e:
                raise RuntimeError(f"Selenium: Ошибка WebDriver при обработке '{path_or_url}'. {e}") from e
            except RuntimeError as e:  # Перехватываем свои ошибки прерывания
                raise e
            except Exception as e:  # Ловим остальные ошибки
                raise RuntimeError(f"Selenium: Непредвиденная ошибка при обработке '{path_or_url}': {e}") from e
            finally:
                if driver:
                    try:
                        driver.quit()
                    except Exception as q_err:
                        log_message(f"Selenium: Ошибка при закрытии WebDriver: {q_err}", level=logging.WARNING)
                    log_message("Selenium: WebDriver закрыт.")
    else:
        raise ValueError(f"Путь не является файлом или URL: {path_or_url}")


# --- Основная логика запуска генерации ---
def generate_thread():
    global folder_data, app, btn_generate, btn_stop

    stop_requested.clear()

    # Блокировка GUI
    try:
        btn_generate.configure(state="disabled")
        if btn_stop and btn_stop.winfo_exists(): btn_stop.configure(state="normal")
        if btn_add_folder and btn_add_folder.winfo_exists(): btn_add_folder.configure(state="disabled")
        for row_data in folder_data:
            if row_data.get('frame') and row_data['frame'].winfo_exists():
                for btn_key in ['edit_button', 'delete_button']:
                    if row_data.get(btn_key) and row_data[btn_key].winfo_exists():
                        row_data[btn_key].configure(state="disabled")
                if row_data.get('quantity_entry') and row_data['quantity_entry'].winfo_exists():
                    row_data['quantity_entry'].configure(state="disabled")
                if row_data.get('checkbox') and row_data['checkbox'].winfo_exists():
                    row_data['checkbox'].configure(state="disabled")
    except Exception as gui_lock_err:
        log_message(f"Ошибка блокировки GUI: {gui_lock_err}", level=logging.WARNING)

    progress.start()
    log_message("--- Начало процесса генерации ---")

    cfg = save_settings()
    if not cfg:
        log_message("Ошибка: Не удалось получить настройки перед генерацией.", level=logging.ERROR)
        # Разблокировка GUI при ошибке настроек
        if progress and progress.winfo_exists(): progress.stop(); progress.set(0)
        if btn_generate and btn_generate.winfo_exists(): btn_generate.configure(state="normal")
        if btn_stop and btn_stop.winfo_exists(): btn_stop.configure(state="disabled")
        if btn_add_folder and btn_add_folder.winfo_exists(): btn_add_folder.configure(state="normal")
        for row_data in folder_data:
            # ... (логика разблокировки папок) ...
            if row_data.get('frame') and row_data['frame'].winfo_exists():
                for btn_key in ['edit_button', 'delete_button']:
                    if row_data.get(btn_key) and row_data[btn_key].winfo_exists(): row_data[btn_key].configure(
                        state="normal")
                if row_data.get('quantity_entry') and row_data['quantity_entry'].winfo_exists(): row_data[
                    'quantity_entry'].configure(state="normal")
                if row_data.get('checkbox') and row_data['checkbox'].winfo_exists(): row_data['checkbox'].configure(
                    state="normal")
        return

    together_key = cfg.get("together_api")
    leonardo_key = cfg.get("leonardo_api")
    main_save_dir = cfg.get("save_dir")
    threads_str = cfg.get("threads", "1")

    errors = []
    if not together_key and not leonardo_key:
        errors.append("Требуется указать Leonardo AI Key или Together AI Key.")
    if together_key and leonardo_key:
        errors.append("Заполните только одну строку API ключа.")
    if not main_save_dir:
        errors.append("Требуется указать основную папку для сохранения.")
    # Проверка существования и доступности папки сохранения
    elif not os.path.isdir(main_save_dir):
        errors.append(f"Папка сохранения '{main_save_dir}' не найдена или не является директорией.")
    elif not os.access(main_save_dir, os.W_OK):
        errors.append(f"Нет прав на запись в папку сохранения '{main_save_dir}'.")

    try:
        threads = int(threads_str)
        if not 1 <= threads <= 30: errors.append("Количество потоков должно быть от 1 до 30.")
    except ValueError:
        errors.append("Количество потоков должно быть числом.");
        threads = 1

    try:
        if errors: raise ValueError("Ошибки ввода:\n" + "\n".join(errors))

        selected_tab = tab_view.get()
        tasks_to_run = []
        task_counter = 0

        # --- Формирование списка задач ---
        if selected_tab == "Обычная генерация":
            log_message("Режим: Обычная генерация")
            default_url = cfg.get("default_image_url")
            default_names_str = cfg.get("default_names", "")
            if not default_url:
                raise ValueError("Обычная генерация: Требуется URL или путь к файлу изображения.")
            names_list = [name.strip() for name in default_names_str.split(',') if name.strip()]
            if not names_list:
                log_message("Используется стандартное имя файла 'generated_image'.", level=logging.WARNING)
                names_list = ["generated_image"]  # Используем список, как ожидает worker

            num_images_to_generate = threads  # Генерируем количество картинок равное числу потоков
            log_message(f"Будет сгенерировано {num_images_to_generate} изображений (по кол-ву потоков).")
            for i in range(num_images_to_generate):
                task_counter += 1
                # Используем базовое имя файла из списка, добавляя суффикс если нужно
                current_name_base = names_list[i % len(names_list)]
                # Добавляем суффикс _<num> только если генерируем больше картинок, чем имен в списке
                name_suffix = f"_{i // len(names_list) + 1}" if num_images_to_generate > len(names_list) and i >= len(
                    names_list) else ""
                current_name = f"{current_name_base}{name_suffix}"

                tasks_to_run.append({
                    'task_id': f"default_{task_counter}",
                    'image_path_or_url': default_url,
                    'together_key': together_key,
                    'leonardo_key': leonardo_key,
                    'save_dir': main_save_dir,  # Сохраняем в основную папку
                    'names_list': [current_name],  # Передаем одно имя для этого задания
                    'is_folder_task': False
                })
            log_message(f"Создано {len(tasks_to_run)} задач для обычной генерации.")

        elif selected_tab == "Сортировка по папкам":
            log_message("Режим: Сортировка по папкам")
            checked_folders_count = 0;
            total_folder_tasks = 0
            folders_config = cfg.get('folders', [])
            for f_data in folders_config:
                if f_data.get('checked', False):
                    checked_folders_count += 1
                    folder_name_val = f_data['name'];
                    folder_url_val = f_data['url']
                    folder_names_str_val = f_data['names'];
                    folder_quantity_val = f_data.get('quantity', 1)

                    if not folder_name_val or not folder_name_val.strip():
                        log_message(f"!!! Пропуск папки с индексом {checked_folders_count}: не указано имя папки.",
                                    level=logging.WARNING);
                        continue
                    folder_name_val = folder_name_val.strip()  # Убираем лишние пробелы

                    if not folder_url_val:
                        log_message(f"!!! Пропуск папки '{folder_name_val}': не указан URL/путь.",
                                    level=logging.WARNING);
                        continue
                    if folder_quantity_val <= 0:
                        log_message(f"!!! Пропуск папки '{folder_name_val}': кол-во должно быть > 0.",
                                    level=logging.WARNING);
                        continue

                    folder_names_list_val = [name.strip() for name in folder_names_str_val.split(',') if name.strip()]
                    if not folder_names_list_val:
                        log_message(f"Папка '{folder_name_val}': не указаны имена, будет использовано имя папки.",
                                    level=logging.WARNING)
                        # Создаем список имен файлов равный количеству картинок
                        folder_names_list_val = [folder_name_val] * folder_quantity_val

                    # Создаем подпапку внутри основной папки сохранения
                    target_folder_dir_val = os.path.join(main_save_dir, folder_name_val)

                    for i in range(folder_quantity_val):
                        task_counter += 1;
                        total_folder_tasks += 1
                        # Выбираем имя для файла из списка имен, циклически
                        current_name_base = folder_names_list_val[i % len(folder_names_list_val)]
                        # Добавляем числовой суффикс, если количество картинок больше числа имен ИЛИ если имя не уникально
                        name_suffix = ""
                        # Проверяем, нужно ли добавлять суффикс (если картинок > имен И текущий индекс >= числа имен)
                        if folder_quantity_val > len(folder_names_list_val) and i >= len(folder_names_list_val):
                            name_suffix = f"_{i // len(folder_names_list_val) + 1}"
                        # ИЛИ если генерируем только одну картинку, но имен несколько (нелогично, но возможно)
                        # elif folder_quantity_val == 1 and len(folder_names_list_val) > 1:
                        # pass # Суффикс не нужен
                        # ИЛИ если имен столько же, сколько картинок (самый частый случай для >1 картинки)
                        # elif folder_quantity_val == len(folder_names_list_val) and folder_quantity_val > 1:
                        # name_suffix = f"_{i + 1}" # Добавляем номер 1, 2, 3...
                        # Упрощенный вариант: добавляем номер, если количество > 1
                        elif folder_quantity_val > 1:
                            name_suffix = f"_{i + 1}"

                        current_name = f"{current_name_base}{name_suffix}"

                        tasks_to_run.append({
                            'task_id': f"folder_{folder_name_val}_{task_counter}",
                            'image_path_or_url': folder_url_val,
                            'together_key': together_key,
                            'leonardo_key': leonardo_key,
                            'save_dir': target_folder_dir_val,  # Путь к подпапке
                            'names_list': [current_name],  # Передаем одно имя для этого задания
                            'is_folder_task': True,
                            'folder_name': folder_name_val
                        })
            log_message(f"Найдено {checked_folders_count} отмеченных папок. Создано {total_folder_tasks} задач.")
            if not tasks_to_run and checked_folders_count > 0:
                raise ValueError(
                    "Сортировка по папкам: В отмеченных папках есть ошибки (URL/кол-во/имя) или не создано ни одной задачи.")
            elif checked_folders_count == 0:
                raise ValueError("Сортировка по папкам: Не выбрано ни одной папки для генерации.")
        else:
            raise ValueError(f"!!! Неизвестная вкладка: {selected_tab}")

        # --- Выполнение задач ---
        results = []
        if tasks_to_run:
            log_message(f"Запуск {len(tasks_to_run)} задач в {threads} потоках...")
            futures_list = []
            with ThreadPoolExecutor(max_workers=threads) as executor:
                # Постановка задач в очередь
                for task_data_item in tasks_to_run:
                    if stop_requested.is_set():
                        log_message("Остановка перед добавлением задач в очередь.", level=logging.WARNING)
                        break
                    futures_list.append(
                        executor.submit(worker, task_data_item['task_id'], copy.deepcopy(task_data_item)))

                completed_count = 0
                failed_count = 0
                cancelled_count = 0
                total_tasks_submitted = len(futures_list)  # Задачи, которые успели попасть в очередь

                # Обработка результатов по мере завершения
                for future in as_completed(futures_list):
                    task_id_from_future = "unknown_task"  # Значение по умолчанию
                    # Попытка получить task_id из аргументов future (не всегда возможно)
                    # Это зависит от реализации as_completed и ThreadPoolExecutor,
                    # обычно прямой доступ к аргументам submit затруднен.
                    # Поэтому будем полагаться на task_id из result.

                    if stop_requested.is_set() and not future.done():
                        # Если нажали стоп и задача еще не выполнена, пытаемся ее отменить
                        if future.cancel():
                            cancelled_count += 1
                            log_message(f"Задача {task_id_from_future} отменена по запросу стоп.",
                                        level=logging.WARNING)
                        # Продолжаем цикл, чтобы обработать уже завершенные или отменить остальные
                        continue

                    try:
                        result = future.result()  # Ждем результат (или исключение)
                        results.append(result)
                        task_id_from_result = result.get('task_id', 'unknown_task')  # Получаем ID из результата

                        if result.get("cancelled"):  # Если worker сам вернул флаг отмены
                            cancelled_count += 1
                            log_message(f"Задача {task_id_from_result} прервана ({cancelled_count} отменено).")
                        elif result.get("success"):
                            completed_count += 1
                            # Обновляем прогресс бар (например, по доле выполненных)
                            progress_value = (completed_count + failed_count + cancelled_count) / total_tasks_submitted
                            if progress and progress.winfo_exists(): progress.set(progress_value)
                            log_message(
                                f"Задача {task_id_from_result} завершена успешно ({completed_count + failed_count + cancelled_count}/{total_tasks_submitted} обработано).")
                        else:  # Если worker вернул success: False без флага cancelled
                            failed_count += 1
                            progress_value = (completed_count + failed_count + cancelled_count) / total_tasks_submitted
                            if progress and progress.winfo_exists(): progress.set(progress_value)
                            log_message(
                                f"Задача {task_id_from_result} завершилась с ошибкой (см. лог выше) ({completed_count + failed_count + cancelled_count}/{total_tasks_submitted} обработано).")

                    except CancelledError:  # Если future.cancel() сработала выше ИЛИ задача была отменена до начала
                        cancelled_count += 1
                        progress_value = (completed_count + failed_count + cancelled_count) / total_tasks_submitted
                        if progress and progress.winfo_exists(): progress.set(progress_value)
                        log_message(
                            f"!!! Задача {task_id_from_future} отменена (CancelledError). ({cancelled_count} отменено)",
                            level=logging.WARNING)
                        results.append({"success": False, "error": "Задача была отменена", "cancelled": True,
                                        "task_id": task_id_from_future})
                    except Exception as exc:  # Ловим исключения, проброшенные из worker (raise)
                        failed_count += 1
                        progress_value = (completed_count + failed_count + cancelled_count) / total_tasks_submitted
                        if progress and progress.winfo_exists(): progress.set(progress_value)
                        # Пытаемся извлечь task_id из исключения, если worker его добавил (не стандартно)
                        task_id_from_exc = getattr(exc, 'task_id', 'unknown_failed_task')
                        log_message(
                            f"!!! Критическая ошибка при получении результата из потока для задачи ~{task_id_from_exc}: {type(exc).__name__} - {exc}",
                            level=logging.CRITICAL)
                        # Логируем traceback только здесь, так как это ошибка выполнения worker
                        logging.exception(f"Traceback критической ошибки потока для задачи ~{task_id_from_exc}:")
                        results.append({"success": False, "error": f"Ошибка потока worker: {exc}", "cancelled": False,
                                        "task_id": task_id_from_exc})

                # После завершения цикла as_completed
                total_processed = completed_count + failed_count + cancelled_count
                unprocessed_tasks = total_tasks_submitted - total_processed  # Задачи, которые могли быть не обработаны из-за break в цикле
                actual_cancelled_or_stopped = cancelled_count + unprocessed_tasks

                log_message(
                    f"--- Обработка задач завершена. Успешно: {completed_count}, Ошибки: {failed_count}, "
                    f"Отменено/Не обработано: {actual_cancelled_or_stopped} (Всего поставлено: {total_tasks_submitted}) ---"
                )

                # Финальное сообщение пользователю
                final_message_title = ""
                final_message_text = ""
                if stop_requested.is_set() or actual_cancelled_or_stopped > 0:
                    final_message_title = "Прервано"
                    final_message_text = f"Генерация прервана/остановлена.\nУспешно: {completed_count}, Ошибки: {failed_count}, Прервано/Не обработано: {actual_cancelled_or_stopped}"
                elif failed_count == 0:
                    final_message_title = "Завершено"
                    final_message_text = f"Генерация ({completed_count} задач) успешно завершена!\nРезультаты сохранены."
                else:
                    final_message_title = "Завершено с ошибками"
                    final_message_text = f"Генерация завершена.\nУспешно: {completed_count}, Ошибки: {failed_count}\nПроверьте лог ({LOG_FILE}) для деталей."

                # Используем app.after для вызова messagebox из основного потока
                if app:
                    app.after(100,
                              lambda title=final_message_title, msg=final_message_text: messagebox.showinfo(title, msg))
                else:
                    print(f"{final_message_title}: {final_message_text}")  # Если GUI уже закрыт

    except ValueError as ve:  # Ошибки валидации ввода
        messagebox.showwarning("Внимание", str(ve))
        log_message(f"--- Генерация прервана (ошибка ввода): {ve} ---", level=logging.WARNING)
    except Exception as e:  # Другие неожиданные ошибки на этапе подготовки
        messagebox.showerror("Критическая ошибка", f"Произошла ошибка перед запуском потоков:\n{type(e).__name__}: {e}")
        log_message(f"!!! КРИТИЧЕСКАЯ ОШИБКА ПОДГОТОВКИ ГЕНЕРАЦИИ:", level=logging.CRITICAL)
        logging.exception("Traceback критической ошибки подготовки:")
    finally:
        # Гарантированная разблокировка GUI
        if progress and progress.winfo_exists():
            progress.stop();
            progress.set(0)

        # Используем app.after для безопасного обновления GUI из потока
        def unlock_gui():
            if btn_generate and btn_generate.winfo_exists(): btn_generate.configure(state="normal")
            if btn_stop and btn_stop.winfo_exists(): btn_stop.configure(state="disabled")
            if btn_add_folder and btn_add_folder.winfo_exists(): btn_add_folder.configure(state="normal")
            for row_data in folder_data:
                if row_data.get('frame') and row_data['frame'].winfo_exists():
                    for btn_key in ['edit_button', 'delete_button']:
                        if row_data.get(btn_key) and row_data[btn_key].winfo_exists(): row_data[btn_key].configure(
                            state="normal")
                    if row_data.get('quantity_entry') and row_data['quantity_entry'].winfo_exists(): row_data[
                        'quantity_entry'].configure(state="normal")
                    if row_data.get('checkbox') and row_data['checkbox'].winfo_exists(): row_data['checkbox'].configure(
                        state="normal")
            log_message("--- Интерфейс разблокирован ---")

        if app: app.after(0, unlock_gui)


def on_generate():
    if not TOGETHER_AVAILABLE:
        messagebox.showerror("Критическая Ошибка",
                             "Библиотека 'together' не найдена.\nУстановите: pip install together")
        log_message("Попытка генерации без библиотеки 'together'.", level=logging.ERROR)
        return
    if btn_generate.cget("state") == "disabled":
        log_message("Генерация уже запущена.", level=logging.WARNING)
        return
    # Запускаем generate_thread в отдельном потоке, чтобы не блокировать GUI
    threading.Thread(target=generate_thread, daemon=True).start()


def stop_generation():
    global btn_generate, btn_stop, progress  # Убрал ненужные app, folder_data
    if not stop_requested.is_set():
        log_message("--- Получен запрос на остановку генерации ---", level=logging.WARNING)
        stop_requested.set()

        # Обновляем GUI немедленно (это безопасно делать из основного потока по клику)
        if btn_generate and btn_generate.winfo_exists():
            btn_generate.configure(
                state="normal")  # Позволяем нажать "Генерировать" снова (хотя finally в generate_thread тоже это сделает)
        if btn_stop and btn_stop.winfo_exists():
            btn_stop.configure(state="disabled")  # Деактивируем кнопку Стоп

        if progress and progress.winfo_exists():
            progress.stop()
            progress.set(0)  # Сбрасываем прогресс
        log_message("--- Сигнал остановки отправлен потокам ---")
    else:
        log_message("Запрос на остановку уже был отправлен.", level=logging.DEBUG)


# --- Функции управления папками ---
def add_folder_row(name="Новая папка", url="", names="", quantity=1, checked_state=True, is_initial=False):
    global folder_rows_frame, folder_data
    if not is_initial:
        # Проверка на дубликат имени папки при добавлении пользователем
        if any(item['name'] == name for item in folder_data if item.get('name')):
            messagebox.showwarning("Дубликат", f"Папка с именем '{name}' уже существует.");
            return None  # Возвращаем None, если папка не добавлена

    row_frame = ctk.CTkFrame(folder_rows_frame, fg_color="transparent");
    row_frame.pack(fill="x", pady=2, padx=5)
    checkbox = ctk.CTkCheckBox(row_frame, text="", width=20);
    checkbox.pack(side="left", padx=(0, 5))
    if checked_state:
        checkbox.select()
    else:
        checkbox.deselect()
    label = ctk.CTkLabel(row_frame, text=name, width=150, anchor="w");
    label.pack(side="left", padx=5)
    url_frame = ctk.CTkFrame(row_frame, fg_color="transparent");
    url_frame.pack(side="left", padx=5, fill="x", expand=True)
    url_entry = ctk.CTkEntry(url_frame, placeholder_text="URL или путь к файлу картинки...");
    url_entry.pack(side="left", fill="x", expand=True, padx=(0, 2))
    if url: url_entry.insert(0, url)
    browse_button = ctk.CTkButton(url_frame, text="...", width=30, command=lambda le=url_entry: browse_file_folder(le));
    browse_button.pack(side="left")
    names_entry = ctk.CTkEntry(row_frame, placeholder_text="Ключевые слова через запятую...", width=200);
    names_entry.pack(side="left", padx=5, fill="x", expand=True)
    if names: names_entry.insert(0, names)
    quantity_frame = ctk.CTkFrame(row_frame, fg_color="transparent");
    quantity_frame.pack(side="left", padx=5)
    ctk.CTkLabel(quantity_frame, text="Кол-во:", width=40).pack(side="left")
    quantity_entry = ctk.CTkEntry(quantity_frame, width=40);
    quantity_entry.pack(side="left");
    quantity_entry.insert(0, str(quantity))
    edit_button = ctk.CTkButton(row_frame, text="✏", width=30,
                                command=lambda rf=row_frame, lbl=label: rename_folder_row(rf, lbl));
    edit_button.pack(side="left", padx=2)
    delete_button = ctk.CTkButton(row_frame, text="🗑", width=30, fg_color="#DB3E3E", hover_color="#B72B2B",
                                  command=lambda rf=row_frame: delete_folder_row(rf));
    delete_button.pack(side="left", padx=(2, 0))

    # Словарь с информацией о виджетах этой строки
    new_folder_gui_info = {
        'url_entry': url_entry,
        'names_entry': names_entry,
        'quantity_entry': quantity_entry,
        'checkbox': checkbox,
        'frame': row_frame,  # Ссылка на сам фрейм строки
        'label': label,  # Ссылка на метку с именем
        'edit_button': edit_button,
        'delete_button': delete_button
    }

    if not is_initial:
        # Добавляем ПОЛНЫЙ словарь (данные + виджеты) в глобальный список folder_data
        folder_data.append({
            'name': name,  # Данные
            'url': url,
            'names': names,
            'quantity': quantity,
            'checked': checked_state,
            **new_folder_gui_info  # Ссылки на виджеты
        })

    # Возвращаем только инфо о виджетах, т.к. при is_initial=True
    # вызывающий код сам добавит это к существующим данным
    return new_folder_gui_info


def delete_folder_row(row_frame):
    global folder_data
    folder_to_remove_idx = -1
    for i, item in enumerate(folder_data):
        if item.get('frame') == row_frame:
            folder_to_remove_idx = i
            break

    if folder_to_remove_idx != -1:
        removed_item = folder_data.pop(folder_to_remove_idx)
        folder_name_val = removed_item.get('name', "Неизвестная папка")
        # Удаляем виджеты строки из GUI
        if row_frame and row_frame.winfo_exists():
            row_frame.destroy()
        log_message(f"Папка '{folder_name_val}' удалена из GUI и данных.")
    else:
        log_message("Не удалось найти данные для удаляемой строки папки в folder_data.", level=logging.WARNING)
        if row_frame and row_frame.winfo_exists():
            row_frame.destroy()
            log_message("Строка папки удалена из GUI (не найдена в данных).", level=logging.DEBUG)


def rename_folder_row(row_frame, name_label_widget):
    global folder_data, app
    current_name_text = name_label_widget.cget("text")

    new_name_text = simpledialog.askstring("Переименовать папку", "Введите новое имя:", initialvalue=current_name_text,
                                           parent=app)

    if new_name_text and new_name_text.strip():
        new_name_text = new_name_text.strip()
        if new_name_text == current_name_text: return

        # Проверка на дубликат по новому имени среди других папок
        if any(item.get('name') == new_name_text for item in folder_data if item.get('frame') != row_frame):
            messagebox.showwarning("Дубликат", f"Папка с именем '{new_name_text}' уже существует.", parent=app)
            return

        # Обновляем текст метки в GUI
        name_label_widget.configure(text=new_name_text)

        # Обновляем поле 'name' в соответствующем словаре в folder_data
        found_in_data = False
        for item in folder_data:
            if item.get('frame') == row_frame:
                item['name'] = new_name_text  # Обновляем данные
                log_message(f"Папка '{current_name_text}' переименована в '{new_name_text}'.")
                found_in_data = True
                break
        if not found_in_data:
            log_message(f"Не удалось найти данные папки '{current_name_text}' при переименовании в folder_data.",
                        level=logging.WARNING)


def clear_all_fields():
    for row_gui_info in folder_data:
        if row_gui_info.get('url_entry') and row_gui_info['url_entry'].winfo_exists():
            row_gui_info['url_entry'].delete(0, END)
        if row_gui_info.get('names_entry') and row_gui_info['names_entry'].winfo_exists():
            row_gui_info['names_entry'].delete(0, END)
    if folder_controls_frame and folder_controls_frame.winfo_exists():
        folder_controls_frame.focus_set()
    log_message("Все поля URL/путей и названий в папках очищены.", level=logging.INFO)


def add_new_folder_prompt():
    global app
    new_name = simpledialog.askstring("Новая папка", "Введите имя для новой папки:", parent=app)
    if new_name and new_name.strip():
        new_name_stripped = new_name.strip()
        # Проверяем на дубликат перед добавлением
        if any(item.get('name') == new_name_stripped for item in folder_data if item.get('name')):
            messagebox.showwarning("Дубликат", f"Папка с именем '{new_name_stripped}' уже существует.", parent=app)
        else:
            add_folder_row(name=new_name_stripped, url="", names="", quantity=1, checked_state=True, is_initial=False)
    elif new_name is not None:  # Если не None и не пустая строка после strip
        messagebox.showwarning("Имя не указано", "Имя папки не может быть пустым.", parent=app)


def on_closing():
    log_message("Завершение работы приложения...")
    stop_requested.set()
    time.sleep(0.2)  # Даем время потокам заметить флаг
    save_settings()  # Сохраняем настройки
    if app: app.destroy()


# --- Настройка GUI ---
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Генератор Изображений (Together Chat → Together Image FLUX.1)")
app.geometry("1000x700")
app.minsize(800, 600)
app.update()
try:
    app.state("zoomed")
except Exception:
    pass  # На некоторых ОС может не работать

app.grid_columnconfigure(0, weight=1)
app.grid_rowconfigure(1, weight=1)  # Для вкладок
app.grid_rowconfigure(2, weight=1)  # Для лога


# --- Код для Ctrl+C/V/X/A (ВАШ КОД) ---
def is_english_layout():
    if os.name != 'nt':
        return True
    try:  # Добавил try-except на случай ошибки ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        thread_id = ctypes.windll.user32.GetWindowThreadProcessId(hwnd, 0)
        hkl = ctypes.windll.user32.GetKeyboardLayout(thread_id)
        langid = hkl & 0xffff
        primary_lang = langid & 0x3ff
        return primary_lang == 0x09
    except Exception as e:
        log_message(f"Ошибка при определении раскладки: {e}", level=logging.WARNING)
        return True  # По умолчанию считаем английской, чтобы не мешать


def _handle_ctrl(event):
    if is_english_layout():
        return

    if not hasattr(event, 'widget') or not isinstance(event.widget, (ctk.CTkEntry, Text, ctk.CTkTextbox)):
        return

    if event.state & 0x4:  # Проверка флага Control
        kc = event.keycode
        if kc == 67:  # C (VK_C)
            event.widget.event_generate('<<Copy>>');
            return 'break'
        if kc == 88:  # X (VK_X)
            event.widget.event_generate('<<Cut>>');
            return 'break'
        if kc == 86:  # V (VK_V)
            event.widget.event_generate('<<Paste>>');
            return 'break'
        if kc == 65:  # A (VK_A)
            event.widget.event_generate('<<SelectAll>>');
            return 'break'


# --- Верхняя часть (API ключ Together, папка сохранения) ---
# ... (GUI код без изменений)...
common_settings_frame = ctk.CTkFrame(app);
common_settings_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
common_settings_frame.grid_columnconfigure(0, weight=1)
common_settings_frame.grid_columnconfigure(1, weight=2)

together_frame = ctk.CTkFrame(common_settings_frame, fg_color="transparent")
together_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
ctk.CTkLabel(together_frame, text="Together AI Key:").pack(side="left", padx=(0, 5))
entry_together = ctk.CTkEntry(together_frame, placeholder_text="Введите ваш ключ от Together AI...", show="*")
entry_together.pack(side="left", fill="x", expand=True)

leonardo_frame = ctk.CTkFrame(common_settings_frame, fg_color="transparent")
leonardo_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
ctk.CTkLabel(leonardo_frame, text="Leonardo AI Key:").pack(side="left", padx=(0, 5))
entry_leonardo = ctk.CTkEntry(leonardo_frame, placeholder_text="Введите ваш ключ от Leonardo AI...", show="*")
entry_leonardo.pack(side="left", fill="x", expand=True)

frame_dir = ctk.CTkFrame(common_settings_frame)
frame_dir.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
frame_dir.grid_columnconfigure(1, weight=1)
ctk.CTkLabel(frame_dir, text="Папка сохранения:").grid(row=0, column=0, padx=(5, 5), pady=5, sticky="w")
entry_save_dir = ctk.CTkEntry(frame_dir, placeholder_text="Выберите папку...")
entry_save_dir.grid(row=0, column=1, padx=(0, 5), pady=5, sticky="ew")
btn_browse_dir = ctk.CTkButton(frame_dir, text="Обзор...", width=80, command=browse_dir)
btn_browse_dir.grid(row=0, column=2, padx=(0, 5), pady=5, sticky="e")

# --- Вкладки ---
tab_view = ctk.CTkTabview(app);
tab_view.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
tab_folders = tab_view.add("Сортировка по папкам")
tab_default = tab_view.add("Обычная генерация")
tab_folders.grid_columnconfigure(0, weight=1);
tab_folders.grid_rowconfigure(1, weight=1)
tab_default.grid_columnconfigure(0, weight=1);
tab_default.grid_rowconfigure(0, weight=1)

# --- Вкладка "Сортировка по папкам" ---
folder_controls_frame = ctk.CTkFrame(tab_folders);
folder_controls_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
btn_add_folder = ctk.CTkButton(folder_controls_frame, text="Добавить папку", command=add_new_folder_prompt,
                               fg_color="green", hover_color="#00A000");
btn_add_folder.pack(side="left", padx=5)
btn_clear_fields = ctk.CTkButton(folder_controls_frame, text="Очистить поля", command=clear_all_fields,
                                 fg_color="#B00000", hover_color="#800000", width=100)
btn_clear_fields.pack(side="left", padx=(0, 5))
folder_rows_frame = ctk.CTkScrollableFrame(tab_folders, label_text="Папки для генерации");
folder_rows_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
folder_rows_frame.grid_columnconfigure(0, weight=1)

# --- Вкладка "Обычная генерация" ---
default_gen_outer_frame = ctk.CTkFrame(tab_default, fg_color="transparent")
default_gen_outer_frame.pack(expand=True, fill="both", padx=10, pady=10)
default_gen_frame = ctk.CTkFrame(default_gen_outer_frame, fg_color="transparent")
default_gen_frame.place(relx=0.5, rely=0.1, anchor="n")
default_gen_frame.grid_columnconfigure(1, weight=1)

ctk.CTkLabel(default_gen_frame, text="Файл/URL источника:").grid(row=0, column=0, pady=(5, 2), padx=5, sticky="w")
frame_url_default = ctk.CTkFrame(default_gen_frame, fg_color="transparent");
frame_url_default.grid(row=0, column=1, padx=5, pady=(0, 5), sticky="ew", columnspan=2)
frame_url_default.grid_columnconfigure(0, weight=1)
entry_url_default = ctk.CTkEntry(frame_url_default, placeholder_text="Укажите путь к файлу или https://...", width=400);
entry_url_default.grid(row=0, column=0, sticky="ew", padx=(0, 5))
btn_browse_file_default = ctk.CTkButton(frame_url_default, text="Обзор...", width=80, command=browse_file_default);
btn_browse_file_default.grid(row=0, column=1)

ctk.CTkLabel(default_gen_frame, text="Названия файлов (через запятую):").grid(row=1, column=0, pady=(5, 2), padx=5,
                                                                              sticky="w")
entry_names_default = ctk.CTkEntry(default_gen_frame, placeholder_text="Пример: apple, banana, cherry", width=400);
entry_names_default.grid(row=1, column=1, padx=5, pady=(0, 10), sticky="ew", columnspan=2)

threads_frame = ctk.CTkFrame(default_gen_frame, fg_color="transparent")
threads_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=(5, 10), sticky="w")
ctk.CTkLabel(threads_frame, text="Потоки (1-30):").pack(side="left", padx=(0, 5))
entry_threads = ctk.CTkEntry(threads_frame, placeholder_text="1", width=50)
entry_threads.pack(side="left")

# --- Область лога ---
log_textbox = ctk.CTkTextbox(app, state=DISABLED, wrap="word", height=150);
log_textbox.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")

# --- Нижняя часть (Прогресс, Кнопки Генерации/Стоп) ---
bottom_frame = ctk.CTkFrame(app);
bottom_frame.grid(row=3, column=0, padx=10, pady=(5, 10), sticky="ew")
bottom_frame.grid_columnconfigure(0, weight=1)

progress = ctk.CTkProgressBar(bottom_frame, orientation="horizontal", mode="indeterminate");
progress.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
progress.set(0);
progress.stop()

btn_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent");
btn_frame.grid(row=0, column=1, padx=10, pady=5, sticky="e")
btn_stop = ctk.CTkButton(btn_frame, text="Стоп", command=stop_generation, height=35, fg_color="#B00000",
                         hover_color="#800000", state=DISABLED)
btn_stop.pack(side="left", padx=(0, 10))
btn_generate = ctk.CTkButton(btn_frame, text="Генерировать", command=on_generate, height=35,
                             font=("Segoe UI", 14, "bold"))
btn_generate.pack(side="left")

# --- Инициализация и Загрузка Настроек ---
log_message("Инициализация приложения...")
# folder_data очищается перед загрузкой, чтобы избежать дублирования
folder_data = []
settings = load_settings()  # Загружает данные в глобальный folder_data

entry_together.insert(0, settings.get("together_api", ""))
entry_leonardo.insert(0, settings.get("leonardo_api", ""))
entry_save_dir.insert(0, settings.get("save_dir", ""))
if entry_threads and entry_threads.winfo_exists():
    entry_threads.insert(0, settings.get("threads", "1"))
else:
    log_message("Ошибка: поле 'entry_threads' не найдено при загрузке настроек.", level=logging.ERROR)
entry_url_default.insert(0, settings.get("default_image_url", ""))
entry_names_default.insert(0, settings.get("default_names", ""))

# Инициализация GUI для папок
if folder_data:
    log_message(f"Создание GUI для {len(folder_data)} загруженных папок...")
    updated_folder_data_list = []
    for f_data_from_settings in folder_data:
        gui_elements_info = add_folder_row(
            name=f_data_from_settings.get('name', 'Unnamed'),
            url=f_data_from_settings.get('url', ''),
            names=f_data_from_settings.get('names', ''),
            quantity=f_data_from_settings.get('quantity', 1),
            checked_state=f_data_from_settings.get('checked', True),
            is_initial=True
        )
        if gui_elements_info:  # Если папка успешно добавлена в GUI
            combined_data = {**f_data_from_settings, **gui_elements_info}
            updated_folder_data_list.append(combined_data)
        else:  # Если произошла ошибка (например, дубликат имени при загрузке - хотя это маловероятно)
            log_message(f"Пропуск папки '{f_data_from_settings.get('name')}' при создании GUI.", level=logging.WARNING)

    folder_data = updated_folder_data_list
else:
    log_message("Настройки папок не найдены. Создание папок по умолчанию...")
    default_folders = ["Main", "App", "Demo", "Login", "Brand 1", "Brand 2", "Brand 3", "About us"]
    # folder_data будет заполняться внутри add_folder_row (is_initial=False)
    for folder_name_default in default_folders:
        add_folder_row(name=folder_name_default, quantity=1, checked_state=True, is_initial=False)

# --- Привязка горячих клавиш (ВАШ КОД) ---
if app:
    app.bind_all('<Control-KeyPress>', _handle_ctrl)

# --- Проверки зависимостей и запуск ---
if not TOGETHER_AVAILABLE:
    # Показываем messagebox только если app уже создано
    if app:
        messagebox.showerror("Критическая Ошибка",
                             "Библиотека 'together' не найдена.\nУстановите: pip install together", parent=app)
    else:
        print("КРИТИЧЕСКАЯ ОШИБКА: Библиотека 'together' не найдена.")
    # sys.exit(1) # Можно раскомментировать для немедленного выхода
if not SELENIUM_AVAILABLE:
    log_message("--- Предупреждение: Selenium не найден, загрузка с некоторых URL может не работать ---",
                level=logging.WARNING)

# Привязываем функцию закрытия окна
app.protocol("WM_DELETE_WINDOW", on_closing)
# Запускаем цикл обновления лога
update_log_textbox()
# Устанавливаем фокус на окно
app.after(100, lambda: app.focus_force())
log_message("Приложение готово к работе.")
# Запускаем главный цикл приложения
app.mainloop()