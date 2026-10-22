"""提供安全的密码哈希与本地配置辅助命令。"""

import hashlib
import hmac
import secrets
import sys
from getpass import getpass

ALGORITHM = "pbkdf2_sha256"
DEFAULT_ITERATIONS = 310_000
MIN_ITERATIONS = 100_000


def hash_password(password: str, iterations: int = DEFAULT_ITERATIONS) -> str:
    if not password:
        raise ValueError("密码不能为空")
    if iterations < MIN_ITERATIONS:
        raise ValueError("迭代次数不能低于安全下限")
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations
    ).hex()
    return f"{ALGORITHM}${iterations}${salt}${digest}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, raw_iterations, salt, expected = encoded.split("$", 3)
        iterations = int(raw_iterations)
        if algorithm != ALGORITHM or iterations < MIN_ITERATIONS:
            return False
        salt_bytes = bytes.fromhex(salt)
        expected_bytes = bytes.fromhex(expected)
    except (AttributeError, TypeError, ValueError):
        return False

    actual = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt_bytes, iterations
    )
    return hmac.compare_digest(actual, expected_bytes)


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("用法：python -m auth.passwords")
    password = getpass("输入密码：")
    confirmation = getpass("再次输入密码：")
    if not password:
        raise SystemExit("密码不能为空")
    if password != confirmation:
        raise SystemExit("两次输入的密码不一致")
    print(hash_password(password))
