import pandas as pd
import df2img
from sklearn.datasets import fetch_openml
from CCPlots.PlotExample import PlotExample
from CCPlots.config import OUTPUT_PATH, COLOR_PALETTE


class MissingDataExample(PlotExample):

    green: str = COLOR_PALETTE['base_colors']['medium_green']
    white: str = COLOR_PALETTE["neutral_colors"]["white"]

    def main(self) -> None:
        # This dataset is from the real world and is also missing some data!
        # https://www.openml.org/search?type=data&status=active&id=1590&sort=runs
        dataset = fetch_openml(data_id=1590, as_frame=True)
        df: pd.DataFrame = dataset.frame  # Convert to Pandas DataFrame

        # Select rows that have naturally missing values
        df_missing_rows = df[df.isnull().any(axis=1)]

        # Check if there are any missing values; otherwise, print a message
        if df_missing_rows.empty:
            print("No naturally missing values found in the dataset.")
            return

        fig = df2img.plot_dataframe(
            # Select max 10 rows with missing data and display only the listed columns
            df_missing_rows.head(10)[["age", "workclass", "sex", "education"]],
            title=dict(
                font_family="Poppins",
                font_size=18,
                text="Examples of missing data in the Adult data set",
            ),
            tbl_header={
                "align": "left",
                "fill_color": self.green,
                "font_color": self.white,
                "font_size": 14,
            },
            fig_size=(1000, 140),
        )

        # Save as an image file
        image_path = OUTPUT_PATH + "naturally_missing_data_table.png"
        df2img.save_dataframe(fig, image_path)

        print(f"Image saved successfully as {image_path}")

if __name__ == "__main__":
    MissingDataExample().main()