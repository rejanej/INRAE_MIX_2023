""" Alexis Alzuria """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
import statsmodels.api as sm

import tkinter as tk
from tkinter import Tk, Button, Frame, Label, Canvas, Scrollbar, Listbox, StringVar, OptionMenu, scrolledtext, filedialog, messagebox, ttk, Entry, LabelFrame

import umap
import hdbscan
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pandasgui import show
import functools


class InterfaceGraphique:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Traitement de Données")

        self.file_path_entry = tk.Entry(self.root, width=50)
        self.file_path_entry.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

        self.browse_button = tk.Button(self.root, text="Parcourir", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=10, pady=10)

        # Listbox pour les feuilles avec sélection multiple
        self.sheet_listbox = tk.Listbox(self.root, selectmode=tk.MULTIPLE, width=30)
        self.sheet_listbox.grid(row=1, column=1, padx=10, pady=10)

        self.sheet_label = tk.Label(self.root, text="Feuilles:")
        self.sheet_label.grid(row=1, column=0, padx=10, pady=10)

        # Listbox pour les variables avec sélection multiple
        self.variables_listbox = tk.Listbox(self.root, selectmode=tk.MULTIPLE, width=30)
        self.variables_listbox.grid(row=2, column=1, padx=10, pady=10)

        self.variables_validate = tk.Button(self.root, text="Choisir", command=lambda : self.update_variables_listbox(selected_sheets_names= self.sheet_listbox.curselection()))
        self.variables_validate.grid(row=1, column=2, padx=10, pady=10)

        self.variables_label = tk.Label(self.root, text="Variables:")
        self.variables_label.grid(row=2, column=0, padx=10, pady=10)

        self.process_button = tk.Button(self.root, text="Traiter", command=self.process_data)
        self.process_button.grid(row=3, column=0, columnspan=3, pady=10)

        self.variables = None
        self.merged_DF = None
        self.selected_sheets = None
        self.root.mainloop()

    def browse_file(self):
        file_path = filedialog.askopenfilename(title="Sélectionnez un fichier")
        self.file_path_entry.delete(0, tk.END)
        self.file_path_entry.insert(0, file_path)

        # Mettre à jour la liste des feuilles dans la Listbox
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        self.sheet_listbox.delete(0, tk.END)
        for sheet_name in sheet_names:
            self.sheet_listbox.insert(tk.END, sheet_name)


    def update_variables_listbox(self, selected_sheets_names):
        self.selected_sheets = selected_sheets_names
        # Mettre à jour la liste des variables dans la Listbox
        self.variables_listbox.delete(0, tk.END)
        print(selected_sheets_names)
        if selected_sheets_names:
            for selected_sheet_name in selected_sheets_names:
                df = pd.read_excel(self.file_path_entry.get(), sheet_name=selected_sheet_name)
                variables = df.columns.tolist()

                for variable in variables:
                    self.variables_listbox.insert(tk.END, variable)

    def process_data(self):
        #selected_sheets = self.sheet_listbox.curselection()
        selected_sheets_names = [self.sheet_listbox.get(index) for index in self.selected_sheets]
        print("Feuilles sélectionnées:", selected_sheets_names)

        selected_variables = self.variables_listbox.curselection()
        selected_variables_names = [self.variables_listbox.get(index) for index in selected_variables]
        print("Variables sélectionnées:", selected_variables_names)

        # Créer un DataFrame en fonction des feuilles et des variables sélectionnées
        dfs = []
        for sheet_name in selected_sheets_names:
            df = pd.read_excel(self.file_path_entry.get(), sheet_name=sheet_name)
            dfs.append(df)

        # Merge les DataFrames en un seul
        self.merged_DF = functools.reduce(lambda left, right: pd.merge(left, right, on='DATE'), dfs)[selected_variables_names]

        print("DataFrame créé avec succès:")
        print(self.merged_DF.head())
        self.variables = self.merged_DF.columns.tolist()
        self.merged_DF.columns
        #données rej
        # df_rej = pd.read_csv("chl_a_stats.csv")
        # df_rej = self.adapt_date(df_rej)
        # #self.merged_DF = pd.merge(self.merged_DF, df_rej, on='DATE')  
        # self.variables = self.merged_DF.columns.tolist()
        self.root.destroy()

    def adapt_date(self, df):
        df['DATE'] = df['File'].str.extract(r'_(\d{8})T(\d{6})_')[0]

        # Ajouter l'heure fixe (12:00:00) à la date
        df['DATE'] = pd.to_datetime(df['DATE'] + ' 12:00:00', format='%Y%m%d %H:%M:%S')
        df = df.sort_values(by='DATE')
        return df
    
    def merge_data(self, DF, variables):
        merged_DF = functools.reduce(lambda left, right: pd.merge(left, right, on='DATE'), DF)      

        #gui = show(merged_DF, show_toolbar=True)
        merged_DF = merged_DF[['DATE']+variables]
            
        # except Exception as e:
        #     tk.messagebox.showerror("Erreur", f"Une erreur s'est produite lors de la fusion des données : {str(e)}")
        merged_DF.to_csv("merged.csv", index=False)
        return merged_DF

    def merge_data_rej(self, DF, variables, df2):
        merged_DF = functools.reduce(lambda left, right: pd.merge(left, right, on='DATE'), DF)      
        merged_DF = pd.merge(merged_DF, df2, on='DATE')  
        print(merged_DF.head())
        merged_DF.to_csv("mergedRej.csv", index=False)

        #gui = show(merged_DF, show_toolbar=True)
        merged_DF = merged_DF[['DATE']+variables]
            
        # except Exception as e:
        #     tk.messagebox.showerror("Erreur", f"Une erreur s'est produite lors de la fusion des données : {str(e)}")
        merged_DF.to_csv("merged.csv", index=False)
        return merged_DF

    def getMergedDF(self):
        return self.merged_DF

    def getVariables(self):
        return self.variables
    
class TraitementsDonnees:
    
    def __init__(self, df, seuil, methode):
        self.df = df
        self.seuil = seuil
        self.methode = methode

    def merge_dataframes(df1, df2, variables):
        """
        Fusionne deux DataFrames en fonction de la colonne 'DATE' et des variables spécifiées.

        Parameters:
        - df1, df2: DataFrames pandas
        - variables: Liste des noms de variables à fusionner

        Returns:
        - DataFrame fusionné
        """
        # Fusionner les DataFrames en fonction de la colonne 'DATE'
        merged_df = pd.merge(df1, df2, on='DATE')

        # Sélectionner les colonnes spécifiées dans la liste 'variables'
        selected_columns = ['DATE'] + variables
        merged_df = merged_df[selected_columns]

        return merged_df


    def calcul_outliers_zscores(self, serie):
        """
        Calcule les indices des outliers en fonction de la série spécifiée.

        Parameters:
        - serie: Série pandas dans laquelle chercher les outliers

        Returns:
        - Indices des outliers 
        """
        z_scores = zscore(self.df[serie])
        seuil = 2
        indices_outliers = np.where(np.abs(z_scores) > self.seuil)[0]

        return indices_outliers
    
    def calcul_outliers_interp(self, colonne):
        """
        Identifie les valeurs aberrantes dans une colonne spécifique en utilisant l'interpolation.

        Parameters:
        - colonne (str): La colonne pour laquelle les valeurs aberrantes doivent être identifiées.

        Returns:
        - indices_outliers (list): Liste des indices des valeurs aberrantes identifiées.
        """
        # Utilisation de la méthode interpolate au lieu des z-scores
        valeurs_interpolées = self.df[colonne].interpolate()

        # Calcul de l'écart entre les valeurs réelles et les valeurs interpolées
        ecarts = abs(self.df[colonne] - valeurs_interpolées)

        # Identification des indices des outliers basés sur un seuil
        indices_outliers = ecarts[ecarts > self.seuil].index.tolist()

        return indices_outliers

    def drop_outliers(self, selected_columns):
        """
        Identifie les valeurs aberrantes dans une colonne spécifique en utilisant l'interpolation.

        Paramètres:
        - colonne (str): La colonne pour laquelle les valeurs aberrantes doivent être identifiées.

        Retourne:
        - indices_outliers (list): Liste des indices des valeurs aberrantes identifiées.
        """
        selcted_df = self.df.copy()
        for i, colonne in enumerate(selected_columns):
            indices_outliers = self.methode(self,colonne)
            selcted_df = selcted_df.drop(index = indices_outliers, errors='ignore')

        return selcted_df
    
    def filter_dataframe_by_date(self, start_date, end_date):
        """
        Filtre le DataFrame par les dates spécifiées.

        Paramètres:
        - start_date (str): Date de début du filtre au format 'YYYY-MM-DD'.
        - end_date (str): Date de fin du filtre au format 'YYYY-MM-DD'.

        Retourne:
        - filtered_df (DataFrame): DataFrame filtré par les dates spécifiées.
        """
        self.df['DATE'] = pd.to_datetime(self.df['DATE'])

        # Filtrer le DataFrame par les dates spécifiées
        filtered_df = self.df[(self.df['DATE'] >= start_date) & (self.df['DATE'] <= end_date)]

        return filtered_df
    
    def display_donnees(self, n_rows, n_cols, selected_columns):
        """
        Affiche des graphiques pour les colonnes spécifiées.

        Paramètres:
        - n_rows (int): Nombre de lignes dans la disposition des sous-graphiques.
        - n_cols (int): Nombre de colonnes dans la disposition des sous-graphiques.
        - selected_columns (list): Liste des colonnes pour lesquelles les graphiques doivent être affichés.

        Retourne:
        - None: Affiche les graphiques directement.
        """
        # Créer les subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.tight_layout(pad=5.0)

        plt.subplots_adjust(hspace=0.5)

        # Parcourir chaque colonne et créer les graphiques
        for i, colonne in enumerate(selected_columns):
            print(colonne)
        # Positionner le subplot
            ax = axes[i // n_cols, i % n_cols] if n_rows > 1 else axes[i % n_cols]

            # Tracer les données
            ax.scatter(self.df.index, self.df[colonne], label='Données', marker='x')

            # Ajouter des étiquettes et un titre
            ax.set_xlabel('Indice')
            ax.set_ylabel(colonne)
            ax.set_title(f'Graphique pour {colonne}')

            # Afficher la légende
            ax.legend()
        # Afficher le graphique principal
        plt.show()

    def display_outliers(self, n_rows, n_cols, selected_columns):
        """
        Affiche des graphiques pour les colonnes spécifiées avec mise en évidence des valeurs aberrantes.

        Paramètres:
        - n_rows (int): Nombre de lignes dans la disposition des sous-graphiques.
        - n_cols (int): Nombre de colonnes dans la disposition des sous-graphiques.
        - selected_columns (list): Liste des colonnes pour lesquelles les graphiques doivent être affichés.

        Retourne:
        - None: Affiche les graphiques directement.
        """
        # Créer les subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.tight_layout(pad=5.0)
        plt.subplots_adjust(hspace=0.5)

        # Parcourir chaque colonne et créer les graphiques
        for i, colonne in enumerate(selected_columns):
            indices_outliers = self.methode(self,colonne)

            # Positionner le subplot
            ax = axes[i // n_cols, i % n_cols] if n_rows > 1 else axes[i % n_cols]

            # Tracer les données
            ax.scatter(self.df.index, self.df[colonne], label='Données', marker='x')

            # Mettre en évidence les outliers
            if len(indices_outliers) > 0:
                ax.scatter(indices_outliers, self.df.iloc[indices_outliers][colonne], color='red', label='Outliers', marker="x")

            # Ajouter des étiquettes et un titre
            ax.set_xlabel('Indice')
            ax.set_ylabel(colonne)
            ax.set_title(f'Graphique pour {colonne}')

            # Afficher la légende
            ax.legend()

        # Afficher le graphique principal
        plt.show()

    def display_outliers_only(self, n_rows, n_cols, selected_columns):
        """
        Affiche des graphiques pour les colonnes spécifiées en ne montrant que les valeurs aberrantes.

        Paramètres:
        - n_rows (int): Nombre de lignes dans la disposition des sous-graphiques.
        - n_cols (int): Nombre de colonnes dans la disposition des sous-graphiques.
        - selected_columns (list): Liste des colonnes pour lesquelles les graphiques doivent être affichés.

        Retourne:
        - None: Affiche les graphiques directement.
        """
        # Créer les subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.tight_layout(pad=5.0)
        plt.subplots_adjust(hspace=0.5)

        # Parcourir chaque colonne et créer les graphiques
        for i, colonne in enumerate(selected_columns):
            # Positionner le subplot
            ax = axes[i // n_cols, i % n_cols] if n_rows > 1 else axes[i % n_cols]
            indices_outliers = self.methode(self,colonne)
            # Tracer les données sans outliers
            #ax.scatter(df.index, df[colonne], label='Données', marker='x')
            
            ax.scatter(indices_outliers, self.df.iloc[indices_outliers][colonne], color='red', label='Outliers', marker="x")

            # Ajouter des étiquettes et un titre
            ax.set_xlabel('Indice')
            ax.set_ylabel(colonne)
            ax.set_title(f'Graphique pour {colonne}')

            # Afficher la légende
            ax.legend()

        # Afficher le graphique principal
        plt.show()

    def remove_outliers(self, n_rows, n_cols, selected_columns):
        """
        Affiche des graphiques pour les colonnes spécifiées en ne montrant que les données sans outliers.

        Paramètres:
        - n_rows (int): Nombre de lignes dans la disposition des sous-graphiques.
        - n_cols (int): Nombre de colonnes dans la disposition des sous-graphiques.
        - selected_columns (list): Liste des colonnes pour lesquelles les graphiques doivent être affichés et les outliers retirés.

        Retourne:
        - None: Affiche les graphiques directement et modifie le DataFrame original.
        """
        # Créer les subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.tight_layout(pad=5.0)
        df2 = self.df.copy()

        # Parcourir chaque colonne et créer les graphiques
        for i, colonne in enumerate(selected_columns):
            # Positionner le subplot
            ax = axes[i // n_cols, i % n_cols] if n_rows > 1 else axes[i % n_cols]
            indices_outliers = self.methode(self,colonne)
            # Tracer les données sans outliers
            #ax.scatter(df.index, df[colonne], label='Données', marker='x')
            data_without_outliers = self.df.drop(index=indices_outliers)
            df2 = df2.drop(index = indices_outliers, errors='ignore')
            ax.scatter(data_without_outliers.index, data_without_outliers[colonne], label='Données sans outliers', marker='x')

            # Ajouter des étiquettes et un titre
            ax.set_xlabel('Indice')
            ax.set_ylabel(colonne)
            ax.set_title(f'Graphique pour {colonne}')

            # Afficher la légende
            ax.legend()

        # Afficher le graphique principal
        plt.show()

    def display_donnees_utiles_sans_outliers(self, n_rows, n_cols, selected_columns):
        """
        Affiche des graphiques pour les colonnes spécifiées en ne montrant que les données sans outliers.

        Paramètres:
        - n_rows (int): Nombre de lignes dans la disposition des sous-graphiques.
        - n_cols (int): Nombre de colonnes dans la disposition des sous-graphiques.
        - selected_columns (list): Liste des colonnes pour lesquelles les graphiques doivent être affichés.

        Retourne:
        - None: Affiche les graphiques directement.
        """
        # Créer les subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.tight_layout(pad=5.0)
        plt.subplots_adjust(hspace=0.5)

        selected_df = self.drop_outliers(selected_columns)
        # Parcourir chaque colonne et créer les graphiques
        for i, colonne in enumerate(selected_columns):
            # Positionner le subplot
            ax = axes[i // n_cols, i % n_cols] if n_rows > 1 else axes[i % n_cols]

            # Tracer les données
            ax.scatter(selected_df.index, selected_df[colonne], label='Données', marker='x')

            # Ajouter des étiquettes et un titre
            ax.set_xlabel('Indice')
            ax.set_ylabel(colonne)
            ax.set_title(f'Graphique pour {colonne}')

            # Afficher la légende
            ax.legend()

        # Afficher le graphique principal
        plt.show()

    def display_courbes_variables_date(self, selected_columns, var_listbox):
        """
        Affiche les courbes pour les variables sélectionnées en fonction de la date.

        Paramètres:
        - selected_columns (list): Liste des colonnes pour lesquelles les courbes doivent être affichées.
        - var_listbox (Listbox): Liste des variables sélectionnées (peut être un widget Listbox de Tkinter).

        Retourne:
        - None: Affiche les courbes directement.
        """
        # Create a subplot
        fig, ax = plt.subplots(figsize=(15, 8))
        selected_variables = var_listbox.curselection()
        selected_variables2 = [selected_columns[i] for i in selected_variables]
        truc = self.df[selected_variables2].values
        # if selected_variables==None:
        #         messagebox.showerror("Erreur", "Veuillez sélectionner une variable.")
        for v in selected_variables:
            c = 0
            for i in range(len(truc)):
                if truc[i,selected_variables.index(v)] == np.max(truc[i,:]) :
                    c+=1
            p = round(c/len(truc)*100,2)
            ax.plot(self.df['DATE'], self.df[selected_columns[v]], label=selected_columns[v]+": "+str(p)+"%")
        # Plot the curve for the selected variable within the date range

        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Values')
        ax.set_title(f'Courbe {selected_columns[v]} pour toute les données')

        # Show legend
        ax.legend()

        # Show the main plot
        plt.show()

    def display_courbes_variables_date_range(self, start_date, end_date, selected_columns, var_listbox):
        """
        Affiche les courbes pour les variables sélectionnées en fonction de la plage de dates spécifiée.

        Paramètres:
        - start_date (str): Date de début de la plage au format 'YYYY-MM-DD'.
        - end_date (str): Date de fin de la plage au format 'YYYY-MM-DD'.
        - selected_columns (list): Liste des colonnes pour lesquelles les courbes doivent être affichées.
        - var_listbox (Listbox): Liste des variables sélectionnées (peut être un widget Listbox de Tkinter).

        Retourne:
        - None: Affiche les courbes directement.
        """

        # Get the start and end dates from the entries
        start_date = start_date_entry.get()
        end_date = end_date_entry.get()
        print(start_date,end_date)
        selected_variables = var_listbox.curselection()
        # Filter the DataFrame based on the selected date range
        df_filtered = self.drop_outliers(selected_columns)
        df_filtered = self.filter_dataframe_by_date(start_date=start_date, end_date=end_date)

        # Create a subplot
        fig, ax = plt.subplots(figsize=(15, 8))
        for v in selected_variables:
            # if selected_variables==None:
            #     messagebox.showerror("Erreur", "Veuillez sélectionner une variable.")
            ax.plot(df_filtered['DATE'], df_filtered[selected_columns[v]], label=selected_columns[v])
        # Plot the curve for the selected variable within the date range

        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Values')
        ax.set_title(f'Courbe de {selected_columns} pour la plage des dates choisie')

        # Show legend
        ax.legend()

        # Show the main plot
        plt.show()

    def display_eventuelles_correlations(self, var_selected1, var_selected2, start_date, end_date, etat_boutton):
        """
        Affiche un nuage de points pour les variables sélectionnées, éventuellement filtrées par une plage de dates,
        avec une courbe de corrélation.

        Paramètres:
        - var_selected1 (StringVar): Variable Tkinter contenant le nom de la première variable sélectionnée.
        - var_selected2 (StringVar): Variable Tkinter contenant le nom de la deuxième variable sélectionnée.
        - start_date (str): Date de début de la plage au format 'YYYY-MM-DD'.
        - end_date (str): Date de fin de la plage au format 'YYYY-MM-DD'.
        - etat_boutton (BooleanVar): Variable Tkinter indiquant l'état d'un bouton (peut être utilisée pour filtrer les données).

        Retourne:
        - None: Affiche le nuage de points avec la courbe de corrélation directement.
        """

        selected_variable1 = var_selected1.get()
        selected_variable2 = var_selected2.get()
        if not etat_boutton.get():
            df_filtered = self.filter_dataframe_by_date(start_date, end_date)
        else:
            df_filtered = self.df

        # Créer un subplot
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Tracer le nuage de points
        ax.scatter(df_filtered[selected_variable1], df_filtered[selected_variable2], label='Nuage de points')

        # Ajuster une ligne de régression linéaire
        slope, intercept = np.polyfit(df_filtered[selected_variable1], df_filtered[selected_variable2], 1)
        min = np.min((df_filtered[selected_variable1].values))
        max = np.max((df_filtered[selected_variable1].values))
        ax.plot(np.linspace(min,max,20),np.linspace(min,max,20),label = "y=x")
        print(slope, intercept)
        ax.plot(df_filtered[selected_variable1], slope * df_filtered[selected_variable1] + intercept, color='red',
                label='Droite de régression')

        # Ajouter des étiquettes et un titre
        ax.set_xlabel(selected_variable1)
        ax.set_ylabel(selected_variable2)
        ax.set_title(f'Nuage de points avec corrélation de {selected_variable2} en fonction de {selected_variable1}')

        # Afficher la légende
        ax.legend()

        # Afficher le graphique principal
        plt.show()

    def display_matrice_correlations(self,selected_columns, start_date, end_date, etat_boutton, var_selected):
        """
        Affiche une matrice de corrélation pour les colonnes sélectionnées, éventuellement filtrées par une plage de dates.

        Paramètres:
        - selected_columns (list): Liste des colonnes pour lesquelles la matrice de corrélation doit être affichée.
        - start_date (str): Date de début de la plage au format 'YYYY-MM-DD'.
        - end_date (str): Date de fin de la plage au format 'YYYY-MM-DD'.
        - etat_boutton (BooleanVar): Variable Tkinter indiquant l'état d'un bouton (peut être utilisée pour filtrer les données).

        Retourne:
        - None: Affiche la matrice de corrélation directement.
        """
        selected_variables = var_selected.curselection()
        selected_variables = [selected_columns[i] for i in selected_variables]
        print(selected_variables)
        df_filtered = self.drop_outliers(selected_variables)
        df_filtered = df_filtered[selected_variables]

        if not etat_boutton.get():
            df_filtered = self.filter_dataframe_by_date(start_date,end_date)

        subset_data = df_filtered[selected_variables]

        # Visualiser la matrice de corrélation
        correlation_matrix = subset_data.corr()

        # Utiliser seaborn pour créer un heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Matrice de corrélation')
        plt.show()

    def display_diag_disp(self,selected_columns, start_date, end_date, etat_boutton, var_selected):
        """
        Affiche des diagrammes de dispersion entre la variable 'ChloroA (µg/l)' et d'autres variables sélectionnées,
        éventuellement filtrées par une plage de dates.

        Paramètres:
        - selected_columns (list): Liste des colonnes pour lesquelles les diagrammes de dispersion doivent être affichés.
        - start_date (str): Date de début de la plage au format 'YYYY-MM-DD'.
        - end_date (str): Date de fin de la plage au format 'YYYY-MM-DD'.
        - etat_boutton (BooleanVar): Variable Tkinter indiquant l'état d'un bouton (peut être utilisée pour filtrer les données).

        Retourne:
        - None: Affiche les diagrammes de dispersion directement.
        """
        selected_variables = var_selected.curselection()
        selected_variables = [selected_columns[i] for i in selected_variables]
        df2 = self.drop_outliers(selected_variables)
        if not etat_boutton.get():
            df_filtered = self.filter_dataframe_by_date(start_date,end_date)
        else:
            df_filtered = df2
        subset_data = df_filtered[selected_variables]

        # Utiliser seaborn pour créer un heatmap
        plt.figure(figsize=(10, 8))
        # Visualiser des diagrammes de dispersion entre ChloroA et les autres variables
        sns.pairplot(subset_data, x_vars=selected_variables, y_vars=['ChloroA (µg/l)'], kind='scatter', height=3)
        plt.suptitle('Diagrammes de dispersion entre ChloroA et les autres variables', y=1.02)
        plt.show()

    def display_cercle_correlation(self,selected_columns, start_date, end_date, etat_boutton, var_selected):
        """
        Affiche un cercle de corrélation pour les colonnes sélectionnées, éventuellement filtrées par une plage de dates.

        Paramètres:
        - selected_columns (list): Liste des colonnes pour lesquelles le cercle de corrélation doit être affiché.
        - start_date (str): Date de début de la plage au format 'YYYY-MM-DD'.
        - end_date (str): Date de fin de la plage au format 'YYYY-MM-DD'.
        - etat_boutton (BooleanVar): Variable Tkinter indiquant l'état d'un bouton (peut être utilisée pour filtrer les données).

        Retourne:
        - None: Affiche le cercle de corrélation directement.
        """
        # Sélectionner les colonnes pertinentes
        selected_variables = var_selected.curselection()
        selected_variables = [selected_columns[i] for i in selected_variables]
        if not etat_boutton.get():
            df_filtered = self.filter_dataframe_by_date(start_date,end_date)
        else:
            df_filtered = self.df

        # selected_variables = ['Hour']+selected_variables
        # df_filtered['Hour'] = pd.to_datetime(df_filtered['DATE']).dt.hour    
        df_filtered = df_filtered[selected_variables]
        df_filtered = df_filtered.dropna()
        # df_day = df_filtered[df_filtered['Hour'].between(6, 18)]  # Supposons que le jour soit de 6h à 18h
        # df_night = df_filtered[~df_filtered['Hour'].between(6, 18)]  # Le reste est considéré comme la nuit
        

        # Standardiser les données
        standardized_data = (df_filtered - df_filtered.mean()) / df_filtered.std()
        #standardized_data = standardized_data.dropna()

        # Appliquer la PCA
        pca = PCA()
        principal_components = pca.fit_transform(standardized_data)

        # Obtenir le cercle de corrélation
        correlation_circle = pca.components_.T * np.sqrt(pca.explained_variance_)

        # Afficher le cercle de corrélation
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        for i in range(len(selected_variables)):
            ax.arrow(0, 0, correlation_circle[i, 0], correlation_circle[i, 1], head_width=0.05, head_length=0.05, fc='red', ec='red')
            ax.text(correlation_circle[i, 0] + 0.1, correlation_circle[i, 1] + 0.1, selected_variables[i], color='black')
        circle = plt.Circle((0, 0), 1, edgecolor='r', facecolor='none', linewidth=2)
        ax.add_patch(circle)
        ax.set_xlabel('PC1 (Composante Principale 1)')
        ax.set_ylabel('PC2 (Composante Principale 2)')
        ax.grid()

        plt.show()

    def display_OLS_Reg(self, selected_columns, start_date, end_date, etat_boutton, var_selected):
        """
        Effectue une régression linéaire ordinaire (OLS) multiple et affiche les résultats à l'aide de la bibliothèque statsmodels.

        Paramètres:
        - selected_columns (list): Liste des colonnes indépendantes pour la régression.
        - start_date (str): Date de début de la plage au format 'YYYY-MM-DD'.
        - end_date (str): Date de fin de la plage au format 'YYYY-MM-DD'.
        - etat_boutton (BooleanVar): Variable Tkinter indiquant l'état d'un bouton (peut être utilisée pour filtrer les données).

        Retourne:
        - None: Affiche les résultats de la régression linéaire dans une fenêtre tkinter.
        """
        selected_variables = var_selected.curselection()
        selected_variables = [selected_columns[i] for i in selected_variables]
        df_selected = self.drop_outliers(selected_variables)
        df_selected = df_selected.dropna()
        if not etat_boutton.get():
            df_selected = self.filter_dataframe_by_date(start_date,end_date)
        # Ajouter une colonne constante pour l'intercept
        X = sm.add_constant(df_selected[selected_variables])
        # Appliquer une transformation logarithmique à la variable dépendante
        y = np.log1p(df_selected['ChloroA (µg/l)'])  # Utilisation de np.log1p pour éviter les problèmes avec les valeurs nulles

        # Effectuer une régression linéaire multiple avec la variable dépendante transformée
        model_log = sm.OLS(y, X).fit()

        # Créer une fenêtre tkinter
        window = tk.Tk()
        window.title("Résultats de la Régression Linéaire")

        # Ajouter un widget de texte déroulant pour afficher les résultats
        text_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=80, height=20)
        text_area.insert(tk.INSERT, model_log.summary())

        # Afficher la fenêtre
        text_area.pack()
        window.mainloop()
    
    def display_cross_correlations(self, selected_columns, variable1, variable2):
        """
        Effectue une régression linéaire ordinaire (OLS) multiple et affiche les résultats à l'aide de la bibliothèque statsmodels.

        Paramètres:
        - selected_columns (list): Liste des colonnes indépendantes pour la régression.
        - start_date (str): Date de début de la plage au format 'YYYY-MM-DD'.
        - end_date (str): Date de fin de la plage au format 'YYYY-MM-DD'.
        - etat_boutton (BooleanVar): Variable Tkinter indiquant l'état d'un bouton (peut être utilisée pour filtrer les données).

        Retourne:
        - None: Affiche les résultats de la régression linéaire dans une fenêtre tkinter.
        """

        selected_variable1 = var_selected1.get()
        selected_variable2 = var_selected2.get()
        # Extraire les colonnes correspondant aux variables sélectionnées
        df2 = self.drop_outliers(selected_columns)

        series1 = df2[selected_variable1]
        series2 = df2[selected_variable2]

        # Calculer la cross-corrélation
        cross_corr = np.correlate(series1, series2, mode='full')
        cross_corr /= np.max(np.abs(cross_corr))

        # Tracer la cross-corrélation
        plt.figure(figsize=(10, 5))
        plt.plot(cross_corr)
        plt.title(f'Cross-correlation between {variable1.get()} and {variable2.get()}')
        plt.xlabel('Lag')
        plt.ylabel('Cross-correlation')
        plt.show()

    def display_UMAP_date(self, selected_columns, n_neighbors, min_dist, start_date, end_date, var_listbox):
        # Filtrer le DataFrame en fonction de la plage de dates sélectionnée
        selected_variables = var_listbox.curselection()

        donnees  = self.filter_dataframe_by_date(start_date,end_date)
        selected_columns = [selected_columns[i] for i in selected_variables]

        # Sélectionner les colonnes pertinentes
        donnees = donnees[selected_columns].values

        # Standardiser les données (optionnel mais souvent recommandé)
        scaler = StandardScaler()
        donnees_standardisees = scaler.fit_transform(donnees)

        # Instancier le modèle UMAP avec les paramètres souhaités
        umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)

        # Appliquer UMAP sur les données
        embedding = umap_model.fit_transform(donnees_standardisees)

        # Créer un DataFrame pour les coordonnées réduites
        df_embedding = pd.DataFrame(data=embedding, columns=['UMAP1', 'UMAP2'])

        # Utiliser K-Means pour obtenir les clusters
        kmeans = KMeans(n_clusters=3)  # ajuster le nombre de clusters selon vos besoins
        df_embedding['Cluster'] = kmeans.fit_predict(df_embedding)

        # Ajouter les colonnes supplémentaires si nécessaire
        # df_embedding['AutreColonne'] = donnees['AutreColonne']

        # Afficher la visualisation UMAP avec des couleurs pour les clusters
        plt.scatter(df_embedding['UMAP1'], df_embedding['UMAP2'], c=df_embedding['Cluster'], cmap='viridis')
        plt.title('Visualisation UMAP avec Clusters')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.colorbar(label='Cluster')
        plt.show()

    def display_UMAP_mois(self, selected_columns, n_neighbors, min_dist, start_date, end_date, etat_boutton, var_listbox):
        selected_variables = var_listbox.curselection()
        selected_columns = [selected_columns[i] for i in selected_variables]
        df_selected = self.drop_outliers(selected_columns)
        df_selected = df_selected.dropna()
        if not etat_boutton.get():
            df_selected = self.filter_dataframe_by_date(start_date,end_date)

        df_selected = df_selected[['DATE']+selected_columns]

        df_selected['Mois'] = pd.to_datetime(df_selected['DATE']).dt.month
        mois_clusters = {}
        for mois in range(1, 13):  # boucle sur les mois de 1 à 12
            mois_donnees = df_selected[df_selected['Mois'] == mois][selected_columns].values
            
            umap_model = umap.UMAP(n_neighbors= n_neighbors, min_dist= min_dist, n_components=2)
            embedding = umap_model.fit_transform(mois_donnees)
            
            mois_clusters[mois] = embedding
        plt.figure(figsize=(12, 8))

        for mois, embedding in mois_clusters.items():
            plt.scatter(embedding[:, 0], embedding[:, 1], label=f'Mois {mois}')

        plt.title('Visualisation UMAP par Mois')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.legend()
        plt.show()

    def display_UMAP_saisons(self, selected_columns, n_neighbors, min_dist, start_date, end_date, etat_boutton, var_listbox):
        selected_variables = var_listbox.curselection()
        selected_columns = [selected_columns[i] for i in selected_variables]
        df_selected = self.drop_outliers(selected_columns)
        df_selected = df_selected.dropna()
        if not etat_boutton.get():
            df_selected = self.filter_dataframe_by_date(start_date,end_date)

        df_selected = df_selected[['DATE']+selected_columns]

        df_selected['Mois'] = pd.to_datetime(df_selected['DATE']).dt.month

        # Mapping des mois aux saisons
        saison_mapping = {1: 'Hiver', 2: 'Hiver', 3: 'Printemps', 4: 'Printemps', 5: 'Printemps', 6: 'Été', 
                        7: 'Été', 8: 'Été', 9: 'Automne', 10: 'Automne', 11: 'Automne', 12: 'Hiver'}
        
        df_selected['Saison'] = df_selected['Mois'].map(saison_mapping)

        saison_clusters = {}

        for saison in df_selected['Saison'].unique():
            saison_donnees = df_selected[df_selected['Saison'] == saison][selected_columns].values
            
            umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)
            embedding = umap_model.fit_transform(saison_donnees)
            
            saison_clusters[saison] = embedding

        plt.figure(figsize=(12, 8))

        for saison, embedding in saison_clusters.items():
            plt.scatter(embedding[:, 0], embedding[:, 1], label=f'{saison}')

        plt.title('Visualisation UMAP par Saison')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.legend()
        plt.show()

    def display_UMAP_saisons_3d(self, selected_columns, n_neighbors,min_dist, start_date, end_date, etat_boutton, var_listbox):
        selected_variables = var_listbox.curselection()
        selected_columns = [selected_columns[i] for i in selected_variables]
        df_selected = self.drop_outliers(selected_columns)

        if not etat_boutton.get():
            df_selected = self.filter_dataframe_by_date(start_date,end_date)
        df_selected = df_selected.dropna()

        df_selected = df_selected[['DATE']+selected_columns]


        df_selected['Mois'] = pd.to_datetime(df_selected['DATE']).dt.month

        saison_mapping = {1: 'Hiver', 2: 'Hiver', 3: 'Printemps', 4: 'Printemps', 5: 'Printemps', 6: 'Été', 
                        7: 'Été', 8: 'Été', 9: 'Automne', 10: 'Automne', 11: 'Automne', 12: 'Hiver'}
        
        df_selected['Saison'] = df_selected['Mois'].map(saison_mapping)

        saison_clusters = {}

        for saison in df_selected['Saison'].unique():
            saison_donnees = df_selected[df_selected['Saison'] == saison][selected_columns].values
            
            umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=3)
            embedding = umap_model.fit_transform(saison_donnees)
            
            saison_clusters[saison] = embedding

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        for saison, embedding in saison_clusters.items():
            ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], label=f'{saison}')

        ax.set_title('Visualisation UMAP par Saison (3D)')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_zlabel('UMAP3')
        ax.legend()
        plt.show()

    def display_HDBSCAN_saisons(self, selected_columns, start_date, end_date, etat_boutton, var_listbox):
        selected_variables = var_listbox.curselection()
        selected_columns = [selected_columns[i] for i in selected_variables]
        df_selected = self.drop_outliers(selected_columns)

        if not etat_boutton.get():
            df_selected = self.filter_dataframe_by_date(start_date,end_date)

        df_selected = df_selected[['DATE']+selected_columns]

        df_selected['Mois'] = pd.to_datetime(df_selected['DATE']).dt.month

        # Mapping des mois aux saisons
        saison_mapping = {1: 'Hiver', 2: 'Hiver', 3: 'Printemps', 4: 'Printemps', 5: 'Printemps', 6: 'Été', 
                        7: 'Été', 8: 'Été', 9: 'Automne', 10: 'Automne', 11: 'Automne', 12: 'Hiver'}
        
        df_selected['Saison'] = df_selected['Mois'].map(saison_mapping)

        saison_clusters = {}

        for saison in df_selected['Saison'].unique():
            saison_donnees = df_selected[df_selected['Saison'] == saison][selected_columns].values
            
            # Utiliser HDBSCAN pour créer les clusters
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
            labels = clusterer.fit_predict(saison_donnees)
            
            # Utiliser UMAP pour la réduction de dimension
            umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
            embedding = umap_model.fit_transform(saison_donnees)
            
            saison_clusters[saison] = (embedding, labels)

        plt.figure(figsize=(12, 8))

        for saison, (embedding, labels) in saison_clusters.items():
            plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, label=f'{saison}')

        plt.title('Visualisation UMAP par Saison avec HDBSCAN')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.legend()
        plt.show()

if __name__ == "__main__":

    def on_checkbox_change(date_range_frame,start_date_entry,start_date_label,end_date_entry,end_date_label):
        # Cette fonction est appelée lorsque l'état du bouton change
        if var.get():
            label.config(text="Sélectionner toutes les données")
            date_range_frame.grid_forget()

        else:
            date_range_frame.grid(row=2, column=0, pady=10)
            start_date_label.grid(row=3, column=0, padx=10)
            start_date_entry.grid(row=3, column=1, padx=10)
            end_date_label.grid(row=3, column=2, padx=10)
            end_date_entry.grid(row=3, column=3, padx=10)
            bouton_afficher_umap_jour.grid(row=4, column=0, pady=10)

            label.config(text="Sélectionner données dans la plage choisie")

    interface =  InterfaceGraphique() #Interface graphique pour choisir les données souhaitées

    df,selected_columns = interface.getMergedDF(),interface.getVariables()
    n_rows, n_cols = len(selected_columns)//2+1,2
    
    data = TraitementsDonnees(df,3, TraitementsDonnees.calcul_outliers_zscores)

    fenetre = Tk()
    fenetre.title("Affichage du graphique")
    style = ttk.Style()
    style.theme_use('clam')  

    var = tk.BooleanVar()

    # Créer un Notebook pour les onglets
    notebook = ttk.Notebook(fenetre)
    notebook.pack(expand=True, fill='both')

    #////////////////////////////////////////////////////////////////////////
    #////                  DEFINITION DES ONGLETS                        ////
    #////////////////////////////////////////////////////////////////////////


    # Onglet pour afficher les données sous forme de nuages de points                   
    onglet_donnees = Frame(notebook)  
    notebook.add(onglet_donnees, text='Visualier Données')

    # Onglet pour afficher les outliers
    onglet_outliers = Frame(notebook)
    notebook.add(onglet_outliers, text='Outliers')

    # Onglet pour afficher les courbes
    onglet_courbes = Frame(notebook)
    notebook.add(onglet_courbes, text='Visualiser Courbes des Variables')

    # Onglet pour l'analyse des corrélations 
    onglet_correlation = Frame(notebook)
    notebook.add(onglet_correlation, text='Visualiser Corrélations')

    # Onglet pour la réduction de dimension 
    onglet_reduction_dim = Frame(notebook)
    notebook.add(onglet_reduction_dim, text='Réduction Dimensionnalité')
                                                                                         
    #////////////////////////////////////////////////////////////////////////
    #////                  DEFINITION DES BOUTONS                        ////
    #////////////////////////////////////////////////////////////////////////       
                                           
    # ONGLET DONNEES: Créer un bouton pour afficher les données
    bouton_display_donnees = tk.Button(onglet_donnees, text="Afficher Données ", command=lambda: data.display_donnees(n_rows=n_rows,n_cols=n_cols, selected_columns=selected_columns))
    bouton_display_donnees.grid(row=0, column=0, pady=10)
    
    # ONGLET OUTLIERS: Créer un bouton pour visualiser les outliers zscores
    bouton_display_outliers = tk.Button(onglet_outliers, text="Afficher Outliers zsocres", command=lambda: data.display_outliers(n_rows=n_rows, n_cols=n_cols, selected_columns=selected_columns))
    bouton_display_outliers.grid(row=0, column=0, pady=10)

    # ONGLET OUTLIERS: Créer un bouton pour visualiser les outliers zscores uniquement
    bouton_display_outliers = tk.Button(onglet_outliers, text="Afficher outliers zscores uniquement", command=lambda : data.display_outliers_only(n_rows=n_rows,n_cols=n_cols,selected_columns=selected_columns))
    bouton_display_outliers.grid(row=1, column=0, pady=10)


    # ONGLET OUTLIERS: Créer un bouton pour visualiser les données sans les outliers zscores 
    bouton_remove_outliers = tk.Button(onglet_outliers, text="Afficher données sans outliers zscores", command=lambda : data.remove_outliers(n_rows=n_rows,n_cols=n_cols,selected_columns=selected_columns))
    bouton_remove_outliers.grid(row=2, column=0, pady=10)
    
    selection_frame_courbes = LabelFrame(onglet_courbes, text="Selection")
    selection_frame_courbes.grid(row=0, column=0, pady=10)

    # ONGLET COURBES: Listbox pour la sélection des variables
    var_listbox_courbes = Listbox(selection_frame_courbes, selectmode='multiple', height=len(selected_columns))
    for variable in selected_columns:
        var_listbox_courbes.insert('end', variable)
    var_listbox_courbes.grid(row=0, column=1, padx=10)

    # ONGLET COURBES: Créer un bouton pour afficher les courbes des variables
    bouton_afficher_courbes = tk.Button(onglet_courbes, text="Afficher Courbes Variables", 
                                                command= lambda:data.display_courbes_variables_date(
                                                            selected_columns=selected_columns,var_listbox=var_listbox_courbes))
    bouton_afficher_courbes.grid(row=1, column=0, pady=10)

    # ONGLET COURBES: Zone de sélection de la date de début et de la date de fin
    date_range_frame = LabelFrame(onglet_courbes, text="Sélection Plage Date")
    date_range_frame.grid(row=2, column=0, pady=10)

    start_date_label = Label(date_range_frame, text="Date Début (YYYY-MM-DD):")
    start_date_label.grid(row=3, column=0, padx=10)
    start_date_entry = Entry(date_range_frame)
    start_date_entry.grid(row=3, column=1, padx=10)

    end_date_label = Label(date_range_frame, text="Date Fin (YYYY-MM-DD):")
    end_date_label.grid(row=3, column=2, padx=10)
    end_date_entry = Entry(date_range_frame)
    end_date_entry.grid(row=3, column=3, padx=10)

    # ONGLET COURBES: Button to trigger the function for date range selection
    button_afficher_courbes_date_range = Button(onglet_courbes, text="Afficher Courbes Variables (avec Date Range)", command=lambda:data.display_courbes_variables_date_range(start_date=start_date_entry.get(),end_date=end_date_entry.get(),selected_columns=selected_columns,var_listbox=var_listbox_courbes))
    button_afficher_courbes_date_range.grid(row=4, column=0, pady=10)

    var_selected1 = StringVar()
    var_selected1.set(selected_columns[0])  

    var_selected2 = StringVar()
    var_selected2.set(selected_columns[0])  

    # Création du bouton Checkbutton
    checkbox = tk.Checkbutton(onglet_correlation, text="Cocher/Décocher", variable=var, command= lambda : on_checkbox_change(date_range_frame=date_range_frame2, start_date_entry=start_date_entry2,start_date_label=start_date_label2,end_date_label=end_date_label2,end_date_entry=end_date_entry2))
    checkbox.grid(row=0, column=0, pady=10)

    # Étiquette pour afficher l'état du bouton
    label = tk.Label(onglet_correlation, text="Bouton décoché")
    label.grid(row=1, column=0, pady=10)

    date_range_frame2 = LabelFrame(onglet_correlation, text="Sélection Plage Date")
    #var.set(not etat_actuel)  # Inverse la valeur de la variable
    # Label and entry for start date
    start_date_label2 = Label(date_range_frame, text="Date Début (YYYY-MM-DD):")
    start_date_label2.grid(row=3, column=0, padx=10)
    start_date_entry2 = Entry(date_range_frame2)
    start_date_entry2.grid(row=3, column=1, padx=10)

    # Label and entry for end date
    end_date_label2 = Label(date_range_frame, text="Date Fin (YYYY-MM-DD):")
    end_date_label2.grid(row=3, column=2, padx=10)
    end_date_entry2 = Entry(date_range_frame2)
    end_date_entry2.grid(row=3, column=3, padx=10)

    selection_frame2 = LabelFrame(onglet_correlation, text="Selection Variables")
    selection_frame2.grid(row=4, column=0, pady=10)



    # Créer la liste déroulante
    dropdown_menu = OptionMenu(onglet_correlation, var_selected1, *selected_columns)
    dropdown_menu.grid(row=5, column=0, pady=10)

    # Créer la liste déroulante
    dropdown_menu = OptionMenu(onglet_correlation, var_selected2, *selected_columns)
    dropdown_menu.grid(row=5, column=1, pady=10)

    # Créer un bouton pour afficher les courbes des variables
    bouton_afficher_ev_corr = tk.Button(onglet_correlation, text="Afficher Données", command=lambda: data.display_eventuelles_correlations(var_selected1=var_selected1,var_selected2=var_selected2, start_date=start_date_entry2.get(), end_date=end_date_entry2.get(), etat_boutton = var))
    bouton_afficher_ev_corr.grid(row=5, column=2, pady=10)

    selection_frame_cor = LabelFrame(onglet_correlation, text="Selection")
    selection_frame_cor.grid(row=6, column=3, pady=10)

    # Create a Listbox for variable selection
    var_listbox_cor = Listbox(selection_frame_cor, selectmode='multiple', height=len(selected_columns))
    for variable in selected_columns:
        var_listbox_cor.insert('end', variable)
    var_listbox_cor.grid(row=6, column=3, padx=10)

    bouton_afficher_mat = tk.Button(onglet_correlation, text="Afficher Matrice Corrélations", command=lambda:data.display_matrice_correlations(selected_columns=selected_columns, start_date=start_date_entry2.get(), end_date=end_date_entry2.get(), etat_boutton = var, var_selected = var_listbox_cor ))
    bouton_afficher_mat.grid(row=6, column=0, pady=10)

    bouton_afficher_diag_disp = tk.Button(onglet_correlation, text="Afficher Diagramme de Dispersion", command=lambda:data.display_diag_disp(selected_columns=selected_columns, start_date=start_date_entry2.get(), end_date=end_date_entry2.get(), etat_boutton = var, var_selected = var_listbox_cor))
    bouton_afficher_diag_disp.grid(row=7, column=0, pady=10)

    # Créer un bouton pour afficher le cercle de corrélation
    bouton_afficher_cercle = tk.Button(onglet_correlation, text="Afficher Cercle de Corrélation", command=lambda:data.display_cercle_correlation(selected_columns=selected_columns, start_date=start_date_entry2.get(), end_date=end_date_entry2.get(), etat_boutton = var, var_selected = var_listbox_cor))
    bouton_afficher_cercle.grid(row=8, column=0, pady=10)

    # Créer un bouton pour afficher les résultats de la corrélation OLS
    bouton_afficher_ols_reg_results = tk.Button(onglet_correlation, text="Afficher OLS Reg", command=lambda:data.display_OLS_Reg(selected_columns=selected_columns, start_date=start_date_entry2.get(), end_date=end_date_entry2.get(), etat_boutton = var, var_selected = var_listbox_cor))
    bouton_afficher_ols_reg_results.grid(row=9, column=0, pady=10)

    selection_frame2 = LabelFrame(onglet_correlation, text="Selection Variables")
    selection_frame2.grid(row=10, column=0, pady=10)

    # Créer la liste déroulante
    dropdown_menu = OptionMenu(onglet_correlation, var_selected1, *selected_columns)
    dropdown_menu.grid(row=11, column=0, pady=10)

    # Créer la liste déroulante
    dropdown_menu = OptionMenu(onglet_correlation, var_selected2, *selected_columns)
    dropdown_menu.grid(row=11, column=1, pady=10)

    # Créer un bouton pour afficher les résultats de la cross corrélation
    bouton_afficher_cross_corr = tk.Button(onglet_correlation, text="Afficher Cross Corrélation Série Temporelle des deux variables", command=lambda: data.display_cross_correlations(selected_columns=selected_columns, variable1=var_selected1,variable2=var_selected2))
    bouton_afficher_cross_corr.grid(row=11, column=2, pady=10)

    # Création du bouton Checkbutton
    checkbox = tk.Checkbutton(onglet_reduction_dim, text="Cocher/Décocher", variable=var, command= lambda : on_checkbox_change(date_range_frame=date_range_frame3,start_date_entry=start_date_entry3,start_date_label=start_date_label3,end_date_label=end_date_label3,end_date_entry=end_date_entry3))
    checkbox.grid(row=0, column=0, pady=10)

    # Étiquette pour afficher l'état du bouton
    label = tk.Label(onglet_correlation, text="Bouton décoché")
    label.grid(row=1, column=0, pady=10)

    date_range_frame3 = LabelFrame(onglet_reduction_dim, text="Sélection Plage Date")

    # Label et entry pour start_date
    start_date_label3 = Label(date_range_frame3, text="Date Début (YYYY-MM-DD):")
    start_date_label3.grid(row=3, column=0, padx=10)
    start_date_entry3 = Entry(date_range_frame3)
    start_date_entry3.grid(row=3, column=1, padx=10)

    # Label et entry pour end_date
    end_date_label3 = Label(date_range_frame3, text="Date Fin (YYYY-MM-DD):")
    end_date_label3.grid(row=3, column=2, padx=10)
    end_date_entry3 = Entry(date_range_frame3)
    end_date_entry3.grid(row=3, column=3, padx=10)


    parameters_frame = LabelFrame(onglet_reduction_dim, text="Sélection des paramètres")
    parameters_frame.grid(row=0, column=1, pady=10)

    # Label et entry pour nb_neighbors
    nb_neighbors_label = Label(parameters_frame, text="nb neighbors")
    nb_neighbors_label.grid(row=0, column=2, padx=10)
    nb_neighbors_entry = Entry(parameters_frame)
    nb_neighbors_entry.grid(row=1, column=2, padx=10)


    # Label et entry pour min_dist
    min_dist_label = Label(parameters_frame, text="min dist")
    min_dist_label.grid(row=0, column=3, padx=10)
    min_dist_entry = Entry(parameters_frame)
    min_dist_entry.grid(row=1, column=3, padx=10)

    selection_frame_umap = LabelFrame(onglet_reduction_dim, text="Selection")
    selection_frame_umap.grid(row=6, column=3, pady=10)

    # Create a Listbox for variable selection
    var_listbox_umap = Listbox(selection_frame_umap, selectmode='multiple', height=len(selected_columns))
    for variable in selected_columns:
        var_listbox_umap.insert('end', variable)
    var_listbox_umap.grid(row=6, column=3, padx=10)

    # Créer un bouton pour afficher les courbes les résultats UMAP pour les mois
    bouton_afficher_umap_jour = tk.Button(onglet_reduction_dim, text="Afficher résultats UMAP jour", command=lambda: data.display_UMAP_date(selected_columns=selected_columns, n_neighbors= int(nb_neighbors_entry.get()), min_dist= float(min_dist_entry.get()), start_date=start_date_entry3.get(), end_date=end_date_entry3.get(), var_listbox=var_listbox_umap))
    bouton_afficher_umap_jour.grid(row=4, column=0, pady=10)

    # Créer un bouton pour afficher les courbes les résultats UMAP pour les mois
    bouton_afficher_umap_mois = tk.Button(onglet_reduction_dim, text="Afficher résultats UMAP/mois", command=lambda: data.display_UMAP_mois(selected_columns=selected_columns, n_neighbors= int(nb_neighbors_entry.get()), min_dist= float(min_dist_entry.get()),start_date=start_date_entry3.get(), end_date=end_date_entry3.get(), etat_boutton = var, var_listbox=var_listbox_umap))
    bouton_afficher_umap_mois.grid(row=5, column=0, pady=10)

    # Créer un bouton pour afficher les résultats UMAP pour les saisons
    bouton_afficher_umap_saisons = tk.Button(onglet_reduction_dim, text="Afficher résultats UMAP/saisons", command=lambda: data.display_UMAP_saisons(selected_columns=selected_columns, n_neighbors= int(nb_neighbors_entry.get()), min_dist= float(min_dist_entry.get()), start_date=start_date_entry3.get(), end_date=end_date_entry3.get(), etat_boutton = var, var_listbox=var_listbox_umap))
    bouton_afficher_umap_saisons.grid(row=6, column=0, pady=10)

    # Créer un bouton pour afficher les résultats UMAP3D pour les saisons
    bouton_afficher_umap_saisons = tk.Button(onglet_reduction_dim, text="Afficher résultats UMAP3D/saisons", command=lambda: data.display_UMAP_saisons_3d(selected_columns=selected_columns, n_neighbors= int(nb_neighbors_entry.get()), min_dist= float(min_dist_entry.get()), start_date=start_date_entry3.get(), end_date=end_date_entry3.get(), etat_boutton=var, var_listbox=var_listbox_umap))
    bouton_afficher_umap_saisons.grid(row=7, column=0, pady=10)

    # Créer un bouton pour afficher les résultats UMAP+HDBSCAN pour les saisons
    bouton_afficher_umap_saisons = tk.Button(onglet_reduction_dim, text="Afficher résultats UMAP+HDBSCAN/saisons", command=lambda: data.display_HDBSCAN_saisons(selected_columns=selected_columns, start_date=start_date_entry3.get(), end_date=end_date_entry3.get(), etat_boutton=var, var_listbox=var_listbox_umap))
    bouton_afficher_umap_saisons.grid(row=8, column=0, pady=10)


    fenetre.mainloop()
