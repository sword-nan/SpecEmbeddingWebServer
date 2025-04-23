import os
import tempfile
from typing import Sequence

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
sns.set_palette("deep")
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.container import StemContainer
from matchms import Spectrum
from rdkit import Chem
from rdkit.Chem import Draw

from type import TokenizerConfig
from data import Tokenizer, TestDataset
from model import SiameseModel
from tester import ModelTester
from utils import top_k_indices, cosine_similarity, read_raw_spectra

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

PAGE_SIZE = 5
BATCH_SIZE = 64
LOADER_BATCH_SIZE = 32
CANDIDATE_PAGE = [2, 5, 10, 20]
SHOW_PROGRESS_BAR = False

device = torch.device("cpu")
tokenizer_config = TokenizerConfig(
    max_len=100,
    show_progress_bar=SHOW_PROGRESS_BAR
)
tokenizer = Tokenizer(100, SHOW_PROGRESS_BAR)
model = SiameseModel(
    embedding_dim=512,
    n_head=16,
    n_layer=4,
    dim_feedward=512,
    dim_target=512,
    feedward_activation="selu"
)
model_state = torch.load("model.ckpt", map_location=device)
model.load_state_dict(model_state)
tester = ModelTester(model, device, SHOW_PROGRESS_BAR)

def custom_stemcontainer(stem_container: StemContainer):
    stem_container.markerline.set_marker("")
    stem_container.baseline.set_color("none")
    stem_container.baseline.set_alpha(0.5)

def draw_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    image = Draw.MolToImage(mol)
    return image

def plot_pair(q: Spectrum, r: Spectrum):
    q_peaks = q.peaks.to_numpy
    r_peaks = r.peaks.to_numpy
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.7), dpi=300)
    ax.text(0.8, 0.8, "query", transform=ax.transAxes)
    ax.text(0.8, 0.2, "reference", transform=ax.transAxes)
    container1 = ax.stem(q_peaks[:, 0], q_peaks[:, 1])
    custom_stemcontainer(container1)
    container2 = ax.stem(r_peaks[:, 0], -r_peaks[:, 1])
    custom_stemcontainer(container2)
    return fig

def generate_result():
    ref_smiles = st.session_state.ref_smiles
    match_indices = st.session_state.match_indices
    df = pd.DataFrame(columns=["ID", "Smiles"])
    for i, index in enumerate(match_indices):
        df.loc[len(df)] = [i + 1, ref_smiles[index]]
    st.session_state.result = df.to_csv(index=False).encode("utf8")

def get_smiles(spectra: Sequence[Spectrum]):
    smiles_seq = [
        s.get("smiles", "")
        for s in spectra
    ]
    return np.array(smiles_seq)

def batch_match(
    progress_bar,
    query_embedding,
    ref_embedding
):
    length = len(query_embedding)
    start_seq, end_seq = gen_start_end_seq(length)
    indices = []

    progress = 0
    for start, end in zip(start_seq, end_seq):
        batch_embedding = query_embedding[start:end]
        cosine_scores = cosine_similarity(batch_embedding, ref_embedding)
        batch_indices = top_k_indices(cosine_scores, 1)
        indices.append(batch_indices)
        if progress + BATCH_SIZE >= length:
            progress = length - 1
        else:
            progress += BATCH_SIZE
        progress_bar.progress((progress + 1) / length)

    return np.concatenate(indices, axis=0)[:, 0]


def init_session_state():
    if "query_path" not in st.session_state:
        st.session_state.query_path = None
    
    if "ref_path" not in st.session_state:
        st.session_state.ref_path = None

    if "data_len" not in st.session_state:
        st.session_state.data_len = None

    if "query_embedding" not in st.session_state:
        st.session_state.query_embedding = None
    
    if "ref_embedding" not in st.session_state:
        st.session_state.ref_embedding = None
    
    if "query_smiles" not in st.session_state:
        st.session_state.query_smiles = None
    
    if "ref_smiles" not in st.session_state:
        st.session_state.ref_smiles = None

    if "query_spectra" not in st.session_state:
        st.session_state.query_spectra = None
    
    if "ref_spectra" not in st.session_state:
        st.session_state.ref_spectra = None
    
    if "match_indices" not in st.session_state:
        st.session_state.match_indices = None

    if "current_page" not in st.session_state:
        st.session_state.current_page = None
    
    if "last_page" not in st.session_state:
        st.session_state.last_page = None
    
    if "page_size" not in st.session_state:
        st.session_state.page_size = PAGE_SIZE

def previous_page():
    current_page = st.session_state.current_page
    if current_page != 1:
        st.session_state.current_page -= 1

def next_page():
    current_page = st.session_state.current_page
    last_page = st.session_state.last_page
    if current_page != last_page:
        st.session_state.current_page += 1

def select_page():
    st.session_state.current_page = int(st.session_state.page_selector)

def set_page_size():
    st.session_state.current_page = 1
    page_size = int(st.session_state.page_size_selector)
    st.session_state.page_size = page_size
    cal_page_num(st.session_state.data_len, page_size)

def cal_page_num(
    length: int,
    page_size: int
):
    page_num, rest = divmod(length, page_size)
    if rest != 0:
        page_num += 1
    st.session_state.last_page = page_num

def gen_start_end_seq(
    length: int,
):
    start_seq = range(0, length, BATCH_SIZE)
    end_seq = range(BATCH_SIZE, length + BATCH_SIZE, BATCH_SIZE)
    return start_seq, end_seq

def embedding(
    progress_bar, 
    tester: ModelTester, 
    tokenizer: Tokenizer,
    spectra: Sequence[Spectrum], 
):
    sequences = tokenizer.tokenize_sequence(spectra)
    start_seq, end_seq = gen_start_end_seq(len(spectra))
    progress = 0
    embedding = []
    for start, end in zip(start_seq, end_seq):
        test_dataset = TestDataset(sequences[start:end])
        test_dataloader = DataLoader(
            test_dataset,
            LOADER_BATCH_SIZE,
            False
        )
        step_embedding = tester.test(test_dataloader)
        if progress + BATCH_SIZE >= len(spectra):
            progress = len(spectra) - 1
        else:
            progress += BATCH_SIZE

        embedding.append(step_embedding)
        progress_bar.progress((progress + 1) / len(spectra))

    embedding = np.concatenate(embedding, axis=0)
    return embedding

def main():
    st.set_page_config(layout="wide")
    st.title("SpecEmbedding")
    tab1, tab2, tab3 = st.tabs(["upload query file", "upload reference/library file", "library match"])

    with tab1:
        st.header("Upload query spectra file(positive mode)")
        query_file = st.file_uploader(
            "upload the query spectra file", 
            type=["msp", "mgf", "mzxml"], 
            key="query_file",
            accept_multiple_files=False
        )
        query_embedding_btn = st.button("Embedding", "query_embedding_btn")
        query_status_box = st.empty()
        if query_embedding_btn:
            if query_file is not None:
                with tempfile.NamedTemporaryFile(delete=True, suffix="." + query_file.name.split(".")[-1]) as tmp_file:
                    tmp_file.write(query_file.getvalue())
                    query_spectra = read_raw_spectra(tmp_file.name)

                progress_bar = st.progress(0, text="Embedding...")
                st.session_state.data_len = len(query_spectra)
                st.session_state.query_spectra = query_spectra
                st.session_state.query_smiles = get_smiles(query_spectra)
                query_embedding = embedding(
                    progress_bar,
                    tester, 
                    tokenizer, 
                    query_spectra, 
                )
                st.session_state.query_embedding = query_embedding
                query_status_box.success("Embedding Success ✅")
            else:
                query_status_box.error("Please upload the spectra file")

    with tab2:
        st.header("Upload reference/library spectra file(positive mode)")
        ref_file = st.file_uploader(
            "upload the reference/library spectra file", 
            type=["msp", "mgf", "mzxml"], 
            key="ref_file", 
            accept_multiple_files=False
        )
        ref_embedding_btn = st.button("Embedding", "ref_embedding_btn")
        ref_status_box = st.empty()
        if ref_embedding_btn:
            if ref_file is not None:
                progress_bar = st.progress(0, text="Embedding...")
                with tempfile.NamedTemporaryFile(delete=True, suffix="." + ref_file.name.split(".")[-1]) as tmp_file:
                    tmp_file.write(ref_file.getvalue())
                    ref_spectra = read_raw_spectra(tmp_file.name)
                
                st.session_state.ref_spectra = ref_spectra
                st.session_state.ref_smiles = get_smiles(ref_spectra)
                ref_embedding = embedding(
                    progress_bar,
                    tester,
                    tokenizer,
                    ref_spectra,
                )
                st.session_state.ref_embedding = ref_embedding
                ref_status_box.success("Embedding Success ✅")
            else:
                ref_status_box.error("Please upload the spectra file")

    with tab3:
        st.header("Start to match")
        launch_btn = st.button("Launch", key="launch_btn")
        match_status_box = st.empty()
        if launch_btn:
            query_embedding = st.session_state.query_embedding
            ref_embedding = st.session_state.ref_embedding
            if query_embedding is None:
                match_status_box.error("No query embedding")
            elif ref_embedding is None:
                match_status_box.error("No reference embedding")
            else:
                progress_bar = st.progress(0, "Match...")
                match_indices = batch_match(progress_bar, query_embedding, ref_embedding)
                st.session_state.match_indices = match_indices
                st.session_state.current_page = 1
                generate_result()
                cal_page_num(st.session_state.data_len, st.session_state.page_size)
                match_status_box.success("match success")

        if st.session_state.match_indices is not None:
            st.subheader(f"Match Result")
            current_page = st.session_state.current_page
            last_page = st.session_state.last_page

            ref_smiles = st.session_state.ref_smiles
            query_spectra = st.session_state.query_spectra
            ref_spectra = st.session_state.ref_spectra
            page_size = st.session_state.page_size

            indices = st.session_state.match_indices
            start = (current_page - 1) * page_size
            end = start + page_size
        
            if current_page == last_page:
                end = indices.shape[0]

            col1, col2, _ = st.columns([1, 1, 5])

            col1.selectbox(
                "page size",
                CANDIDATE_PAGE,
                key="page_size_selector",
                disabled=False,
                label_visibility="collapsed",
                index=CANDIDATE_PAGE.index(page_size),
                on_change=set_page_size,
            )

            col2.download_button(
                label="download result",
                data=st.session_state.result,
                file_name="data.csv",
                mime="text/csv"
            )

            pre_btn, current, next_btn, page_selector, _ =  st.columns([1, 1, 1, 1, 2])
            pre_btn.button("previous page", key="pre_btn", on_click=previous_page)
            current.subheader(f"current page: {current_page}")
            next_btn.button("next page", key="next_btn", on_click=next_page)
            page_selector.selectbox(
                label="target page",
                key="page_selector",
                options=range(1, last_page + 1),
                disabled=False,
                index=current_page - 1,
                label_visibility="collapsed",
                on_change=select_page,
            )

            col1, col2, col3, col4 = st.columns([1, 4, 6, 4])
            col1.subheader("Index")
            col2.subheader("Smiles")
            col3.subheader("MS/MS Spectra Pair")
            col4.subheader("Molecular Structure")

            for i in range(start, end):
                query_index = i
                ref_index = indices[i]
                id_label, smiles_label, pair_viewer, mol_viewer = st.columns([2, 4, 6, 4])
                id_label.subheader(i + 1)
                smiles_label.text(ref_smiles[ref_index])
                pair_fig = plot_pair(query_spectra[query_index], ref_spectra[ref_index])
                pair_viewer.pyplot(pair_fig, use_container_width=True)
                mol_image = draw_mol(ref_smiles[ref_index])
                mol_viewer.image(mol_image, use_container_width=True)

if __name__ == "__main__":
    init_session_state()
    main()