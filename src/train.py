from src.preprocessing import prepare_data

def main():
    train_df, val_df, test_df, csv_path, class_to_label = prepare_data()

    print("CSV salvo em:", csv_path)
    print("Mapa de classes:", class_to_label)

if __name__ == "__main__":
    main()
