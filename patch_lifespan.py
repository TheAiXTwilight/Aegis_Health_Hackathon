from pathlib import Path

path = Path('/home/user/backend/main.py')
content = path.read_text(encoding='utf-8')

old = '''    init_db()
    pipeline = AegisPipeline()'''

new = '''    init_db()
    if settings.AEGIS_SEED_DEMO_USERS:
        from app.db.session import SessionLocal
        db = SessionLocal()
        try:
            seed_demo_users(db)
        finally:
            db.close()
    pipeline = AegisPipeline()'''

if old not in content:
    raise SystemExit('lifespan pattern not found')

content = content.replace(old, new, 1)
path.write_text(content, encoding='utf-8')
print('patched lifespan')
