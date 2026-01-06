import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders.parsers.language.language_parser import (
    LANGUAGE_SEGMENTERS,
)
from langchain_community.document_loaders.parsers.language.language_parser import (
    Language as LanguageLiteral,
)
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


@dataclass
class FileConfig:
    glob: str
    lang: Language


DocCollection: TypeAlias = dict[Language, list[Document]]


# Files types to be loaded from base directory:
FILE_CONFIGS = [
    FileConfig("**/*.py", Language.PYTHON),
    FileConfig("**/*.md", Language.MARKDOWN),
    FileConfig("**/*.rst", Language.RST),
    FileConfig("**/*.html", Language.HTML),
]


class Preprocessor(ABC):
    @abstractmethod
    def get_splits(
        self, chunk_size: int, chunk_overlap: int, tmp_dir: Path
    ) -> list[Document]: ...

    def _load_docs(self, path: Path) -> DocCollection:
        doc_collection: DocCollection = defaultdict(list)

        for config in FILE_CONFIGS:
            if config.lang in LANGUAGE_SEGMENTERS:
                loader = GenericLoader.from_filesystem(
                    path=path,
                    glob=config.glob,
                    parser=LanguageParser(language=cast(LanguageLiteral, config.lang)),
                )
            else:
                loader = DirectoryLoader(
                    path=str(path),
                    glob=config.glob,
                    loader_cls=TextLoader,
                    loader_kwargs={"autodetect_encoding": True},
                )

            docs = loader.load()
            doc_collection[config.lang].extend(docs)

        return doc_collection

    def _split_docs(
        self, doc_collection: DocCollection, chunk_size: int, chunk_overlap: int
    ):
        splits: list[Document] = []
        for lang, docs in doc_collection.items():
            splitter = RecursiveCharacterTextSplitter.from_language(
                lang, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

            splits.extend(splitter.split_documents(docs))

        return splits


class GitRepoPreprocessor(Preprocessor):
    def __init__(self, repo_url: str, branch: str | None = None) -> None:
        super().__init__()
        self._repo_url = repo_url
        self._branch = branch

    def get_splits(
        self, chunk_size: int, chunk_overlap: int, tmp_dir: Path
    ) -> list[Document]:
        if self._branch is not None:
            cmd = [
                "git",
                "clone",
                "--branch",
                self._branch,
                "--single-branch",
                self._repo_url,
                str(tmp_dir),
            ]
        else:
            cmd = ["git", "clone", "--single-branch", self._repo_url, str(tmp_dir)]

        subprocess.call(cmd)

        docs = self._load_docs(tmp_dir)
        splits = self._split_docs(docs, chunk_size, chunk_overlap)

        return splits
