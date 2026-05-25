"""TreeEnvironment：单棵视频树的运行时环境。

提供节点查询、字幕获取、帧路径解析和语义检索能力。
embedding 模型和向量索引延迟加载，首次 search_similar 时构建并缓存到 npz。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from core.tree.summarizer import (
    summarize_children,
    summarize_node,
    summarize_nodes_batch,
)

_SUMMARY_FIELDS = {1: "scene_summary", 2: "event_description", 3: "frame_summary"}


class TreeEnvironment:
    """单棵视频树的运行时环境，提供节点查询和语义检索。

    参数:
        tree_json_path: 树 JSON 文件路径。
        tool_client: 工具用 LLMClient（thinking=False），用于摘要。
        prompts_dir: prompt 文件目录，view_node / search_similar 需要。
        embedding_model: sentence-transformers 模型名称。
    """

    def __init__(
        self,
        tree_json_path: str | Path,
        tool_client: Any,
        prompts_dir: str | Path | None = None,
        embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
    ) -> None:
        path = Path(tree_json_path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self._nodes: dict[str, dict[str, Any]] = data["nodes"]
        self._video_id: str = data["videoID"]
        self._duration_category: str = data["duration_category"]
        self._duration_seconds: float = data["duration_seconds"]
        self._domain: str = data["domain"]
        self._tree_dir: Path = path.parent
        self._tool_client = tool_client
        self._prompts_dir: Path | None = Path(prompts_dir) if prompts_dir else None
        self._embedding_model_name: str = embedding_model
        self._chunk_node_ids: list[str] | None = None
        self._embeddings: np.ndarray | None = None
        self._model: Any = None

    # ------------------------------------------------------------------
    # 内部文本提取
    # ------------------------------------------------------------------

    def _extract_card_text(self, card: dict[str, Any]) -> str:
        """递归提取 card 中所有字符串值，拼接为单一文本。

        参数:
            card: 节点 card 字典。

        返回:
            拼接后的文本。
        """
        texts: list[str] = []
        self._collect_strings(card, texts)
        return "\n".join(texts)

    def _collect_strings(self, obj: object, texts: list[str]) -> None:
        """递归收集对象中的所有非空字符串。

        参数:
            obj: 任意嵌套结构（dict / list / str / None）。
            texts: 收集结果列表（原地修改）。
        """
        if isinstance(obj, str):
            stripped = obj.strip()
            if stripped:
                texts.append(stripped)
        elif isinstance(obj, dict):
            for v in obj.values():
                self._collect_strings(v, texts)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_strings(item, texts)

    def _get_summary(self, node: dict[str, Any]) -> str:
        """获取节点摘要文本（含字幕，不截断）。

        参数:
            node: 节点字典。

        返回:
            摘要字符串。
        """
        field = _SUMMARY_FIELDS.get(node["level"])
        text = node["card"].get(field, "") if field else ""
        if not text:
            text = self._extract_card_text(node["card"])
        subtitle = node.get("subtitle", "")
        if subtitle:
            text += f" | 字幕: {subtitle}"
        return text

    def _node_full_text(self, node: dict[str, Any]) -> str:
        """获取节点完整文本（card + subtitle）。

        参数:
            node: 节点字典。

        返回:
            拼接后的全文本。
        """
        card_text = self._extract_card_text(node["card"])
        subtitle = node.get("subtitle", "")
        if subtitle:
            return f"{card_text}\n{subtitle}"
        return card_text

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 4000, overlap: int = 800) -> list[str]:
        """将长文本切分为重叠 chunk。

        参数:
            text: 原始文本。
            chunk_size: 每个 chunk 的最大字符数。
            overlap: 相邻 chunk 的重叠字符数。

        返回:
            chunk 列表（短文本直接返回单元素列表）。
        """
        if len(text) <= chunk_size:
            return [text]
        chunks: list[str] = []
        step = chunk_size - overlap
        for start in range(0, len(text), step):
            chunk = text[start : start + chunk_size]
            chunks.append(chunk)
            if start + chunk_size >= len(text):
                break
        return chunks

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def get_subtitle(self, node_id: str) -> str:
        """获取节点字幕文本。

        参数:
            node_id: 节点 ID。

        返回:
            字幕文本，无字幕时返回空字符串。
        """
        node = self._nodes.get(node_id)
        if node is None:
            return ""
        return node.get("subtitle", "")

    def resolve_frame_paths(self, node_ids: list[str]) -> list[Path]:
        """解析节点 ID 列表为帧文件路径。

        支持 L3 节点（直接映射）和 L2 节点（展开为全部 L3 子节点）。

        参数:
            node_ids: 节点 ID 列表。L3 模式 1-4 个，L2 模式仅 1 个，不可混合。

        返回:
            帧文件 Path 列表，顺序与展开后的 L3 节点一致。

        异常:
            ValueError: 参数违反约束（空列表、L1 节点、混合传入、数量超限）。
            KeyError: 节点不存在。
        """
        if not node_ids:
            raise ValueError("node_ids 不能为空")

        levels: set[int] = set()
        for nid in node_ids:
            node = self._nodes.get(nid)
            if node is None:
                raise KeyError(f"节点不存在: {nid}")
            level = node["level"]
            if level == 1:
                raise ValueError(f"不支持 L1 节点: {nid}")
            levels.add(level)

        if len(levels) > 1:
            raise ValueError("L2 和 L3 节点不能混合传入")

        target_level = levels.pop()

        if target_level == 2:
            if len(node_ids) > 1:
                raise ValueError("最多传入 1 个 L2 节点")
            l3_ids = self._nodes[node_ids[0]]["children_ids"]
        else:
            if len(node_ids) > 4:
                raise ValueError("最多传入 4 个 L3 节点")
            l3_ids = node_ids

        frames_dir = self._tree_dir / "frames"
        paths: list[Path] = []
        for l3_id in l3_ids:
            suffix = l3_id.replace(f"{self._video_id}_", "")
            paths.append(frames_dir / f"{suffix}.jpg")
        return paths

    def view_node(self, node_id: str, question: str) -> str:
        """查看节点信息：question-conditioned 摘要 + 子节点概览。

        参数:
            node_id: 节点 ID。
            question: Agent 当前关注的具体问题。

        返回:
            包含 LLM 摘要和子节点概览的格式化文本。

        异常:
            KeyError: 节点不存在。
        """
        assert self._prompts_dir is not None, "view_node 需要 prompts_dir"

        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"节点不存在: {node_id}")

        t_start, t_end = node["time_range"]
        level_name = {1: "场景层", 2: "事件层", 3: "关键帧层"}.get(
            node["level"], "未知"
        )

        raw_text = self._node_full_text(node)
        summary = summarize_node(
            self._tool_client, raw_text, question, self._prompts_dir
        )

        parts = [
            f"[节点] {node_id} | {level_name} | {t_start:.1f}-{t_end:.1f}s",
            "",
            summary,
        ]

        children_ids = node.get("children_ids", [])
        if children_ids:
            children_info = []
            for cid in children_ids:
                child = self._nodes[cid]
                field = _SUMMARY_FIELDS.get(child["level"])
                child_summary = child["card"].get(field, "") if field else ""
                if not child_summary:
                    child_summary = self._extract_card_text(child["card"])[:120]
                children_info.append(
                    {
                        "id": cid,
                        "time_range": child["time_range"],
                        "summary": child_summary,
                    }
                )
            children_overview = summarize_children(
                self._tool_client, children_info, question, self._prompts_dir
            )
            parts.append(
                f"\n[子节点概览] {len(children_ids)} 个子节点\n{children_overview}"
            )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 语义检索
    # ------------------------------------------------------------------

    def _get_model(self) -> Any:
        """延迟加载 embedding 模型。

        返回:
            SentenceTransformer 模型实例。
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._embedding_model_name, trust_remote_code=True
            )
        return self._model

    def _ensure_embeddings(self) -> None:
        """延迟加载或生成 embedding 索引（支持 chunking）。"""
        if self._embeddings is not None:
            return

        cache_path = self._tree_dir / f"{self._video_id}.embeddings.npz"

        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            self._chunk_node_ids = data["chunk_node_ids"].tolist()
            self._embeddings = data["embeddings"]
            return

        chunk_node_ids: list[str] = []
        texts: list[str] = []
        for nid, node in self._nodes.items():
            full_text = self._node_full_text(node)
            for chunk in self._chunk_text(full_text):
                chunk_node_ids.append(nid)
                texts.append(chunk)

        self._chunk_node_ids = chunk_node_ids

        model = self._get_model()
        self._embeddings = model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        np.savez(
            cache_path,
            chunk_node_ids=np.array(self._chunk_node_ids),
            embeddings=self._embeddings,
        )

    def search_similar(self, query: str, question: str, k: int = 5) -> str:
        """语义检索 top-k 节点，返回 question-conditioned 摘要。

        参数:
            query: 搜索关键词。
            question: Agent 当前关注的具体问题。
            k: 返回数量。

        返回:
            格式化的检索结果文本，每条含 LLM 摘要。
        """
        assert self._prompts_dir is not None, "search_similar 需要 prompts_dir"

        self._ensure_embeddings()

        model = self._get_model()
        query_emb = model.encode([query], normalize_embeddings=True)
        scores = (self._embeddings @ query_emb.T).squeeze()

        best: dict[str, float] = {}
        for idx, score in enumerate(scores):
            nid = self._chunk_node_ids[idx]
            if nid not in best or float(score) > best[nid]:
                best[nid] = float(score)

        sorted_all = sorted(best.items(), key=lambda x: x[1], reverse=True)

        deduped: list[tuple[str, float]] = []
        seen_prefixes: set[str] = set()
        for nid, score in sorted_all:
            is_ancestor_of_seen = any(s.startswith(nid + "_") for s in seen_prefixes)
            if is_ancestor_of_seen:
                continue
            deduped.append((nid, score))
            seen_prefixes.add(nid)
            if len(deduped) >= k:
                break

        items = []
        for nid, score in deduped:
            node = self._nodes[nid]
            raw_text = self._node_full_text(node)
            t_start, t_end = node["time_range"]
            extra = (
                f"L{node['level']}  score={score:.4f}  [{t_start:.1f}s, {t_end:.1f}s]"
            )
            items.append((nid, raw_text, extra))

        summaries = summarize_nodes_batch(
            self._tool_client, items, question, self._prompts_dir
        )

        lines = []
        for i, (nid, summary_text) in enumerate(summaries):
            _, _, extra = items[i]
            lines.append(f"{i + 1}. {nid} | {extra}\n   {summary_text}")

        header = f'[搜索结果] 查询 "{query}" → {len(deduped)} 个相关节点'
        return header + "\n\n" + "\n\n".join(lines)
