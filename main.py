from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


def main():
    print("Hello from hp-cccb!")


if __name__ == "__main__":
    main()
