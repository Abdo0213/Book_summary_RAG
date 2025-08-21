
@app.get("/download/english")
def download_english():
    filepath = os.path.join(SUMMARY_DIR, "summary_en.pdf")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="English summary not found")
    return FileResponse(filepath, media_type="application/pdf", filename="summary_en.pdf")

@app.get("/download/arabic")
def download_arabic():
    filepath = os.path.join(SUMMARY_DIR, "summary_ar.pdf")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Arabic summary not found")
    return FileResponse(filepath, media_type="application/pdf", filename="summary_ar.pdf")



@app.post("/register")
def register(user: User):
    conn = get_conn()
    cur = conn.cursor()
    hashed_pw = pwd_context.hash(user.password)
    try:
        cur.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                    (user.username, user.email, hashed_pw))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    finally:
        conn.close()
    return {"msg": "User registered"}

@app.post("/login")
def login(user: User):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, password FROM users WHERE email=?", (user.email,))
    row = cur.fetchone()
    conn.close()
    if not row or not pwd_context.verify(user.password, row[1]):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return {"msg": "Login successful", "user_id": row[0]}

# --- Utility: create pdf ---
def create_pdf(text, filename):
    filepath = os.path.join(SUMMARY_DIR, filename)
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)

    # split long text into lines
    y = height - 50
    for line in text.split("\n"):
        c.drawString(50, y, line)
        y -= 20
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 50
    c.save()
    return filepath


