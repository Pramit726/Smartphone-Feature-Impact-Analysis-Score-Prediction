import base64
import io
import pandas as pd
import matplotlib.pyplot as plt


def make_columns_readable(columns: pd.Index) -> pd.Index:
    """
    Transforms column names to be more readable for end-users.

    :param columns: The original column index (typically from a Pandas DataFrame).
    :type columns: pd.Index
    :return: A new column index with readable names.
    :rtype: pd.Index
    """

    readable_columns = []
    for col in columns:
        col = col.replace("standard_scaling__", "")
        col = col.replace("minmax_scaling__", "")
        col = col.replace("nominal_encoding__", "")
        col = col.replace("ordinal_encoding__", "")
        col = col.replace("_", " ")

        # Specific changes to make it more readable
        col = col.replace("brand name", "Brand")
        col = col.replace("processor brand", "Processor")
        col = col.replace("connectivity features", "Connectivity")
        col = col.replace("storage expandability", "Storage Expandable")
        col = col.replace("fast charging available", "Fast Charging")
        col = col.replace("extended memory available", "Memory Expandable")
        col = col.replace("num cores", "Cores")
        col = col.replace("ram capacity", "RAM")
        col = col.replace("internal memory", "Internal Storage")
        col = col.replace("fast charging", "Fast Charging (Watts)")
        col = col.replace("refresh rate", "Refresh Rate (Hz)")
        col = col.replace("num rear cameras", "Rear Cameras")
        col = col.replace("num front cameras", "Front Cameras")
        col = col.replace("primary camera rear", "Rear Camera (MP)")
        col = col.replace("primary camera front", "Front Camera (MP)")
        col = col.replace("extended upto", "Expandable Memory (GB)")
        col = col.replace("performance score", "Performance Score")
        col = col.replace("camera quality", "Camera Quality")
        col = col.replace("screen size", "Screen Size (inches)")
        col = col.replace("resolution", "Resolution")
        col = col.replace("has 5g", "5G")
        col = col.replace("has nfc", "NFC")
        col = col.replace("has ir blaster", "IR Blaster")
        col = col.replace("apple", "Apple")
        col = col.replace("infinix", "Infinix")
        col = col.replace("iqoo", "iQOO")
        col = col.replace("motorola", "Motorola")
        col = col.replace("oneplus", "OnePlus")
        col = col.replace("oppo", "Oppo")
        col = col.replace("poco", "Poco")
        col = col.replace("realme", "Realme")
        col = col.replace("samsung", "Samsung")
        col = col.replace("tecno", "Tecno")
        col = col.replace("vivo", "Vivo")
        col = col.replace("xiaomi", "Xiaomi")
        col = col.replace("bionic", "Bionic")
        col = col.replace("dimensity", "Dimensity")
        col = col.replace("exynos", "Exynos")
        col = col.replace("helio", "Helio")
        col = col.replace("snapdragon", "Snapdragon")
        col = col.replace("tiger", "Tiger")
        col = col.replace("unisoc", "Unisoc")
        col = col.replace("other", "Other")

        readable_columns.append(col)

    return pd.Index(readable_columns)


def encode_fig(fig: plt.Figure) -> str:
    """
    Encodes a matplotlib figure as a base64 string.

    :param fig: The matplotlib figure to encode.
    :type fig: plt.Figure
    :return: The encoded figure as a base64 string.
    :rtype: str
    """

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return img_base64
