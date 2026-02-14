from preprocessing import prepare_data

if __name__ == "__main__":
    full_df, path = prepare_data()
    print("Preprocessing finalizado.")
    print("CSV salvo em:", path)
