//! Bounded local work for importing the edges of one cloud node.
//!
//! A pull is not an initiative-wide reconciliation. Only edges incident to
//! the imported node can have acquired a newly available endpoint. Endpoint
//! existence is checked in batches, without rendering bodies or resolving names.

use std::collections::{BTreeMap, HashSet};

use cozo::{DataValue, ScriptMutability};

use crate::{EdgeType, Result, Store, link_with_weight};

/// Outcome of a node-local edge import. Existing local decisions win.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct EdgeImportResult {
    pub linked: usize,
    pub kept: usize,
}

/// The cloud representation of an edge; unknown future edge types are skipped.
pub type CloudEdge = (String, String, String, f64);

/// Restore only `target`'s missing edges, retaining local weights AND local
/// retractions. A repeated pull must not undo an explicit local unlink.
/// Network access belongs to adapters; every substrate access stays here.
/// The scope guard is held for one mutation, never for the entire import.
pub fn restore_cloud_edges(
    store: &Store,
    initiative: &str,
    target: &str,
    edges: &[CloudEdge],
) -> Result<EdgeImportResult> {
    restore_cloud_edges_cancellable(store, initiative, target, edges, || false)
}

/// Cooperative cancellation is checked before reads and between mutations.
/// A running Cozo query is not preempted; already imported data stays valid.
pub fn restore_cloud_edges_cancellable(
    store: &Store,
    initiative: &str,
    target: &str,
    edges: &[CloudEdge],
    cancelled: impl Fn() -> bool,
) -> Result<EdgeImportResult> {
    if cancelled() {
        return Err(crate::Error::Invalid(
            "edge import cancelled; node may already be local".into(),
        ));
    }
    let mut candidates = Vec::new();
    let mut endpoints = HashSet::new();
    let mut seen = HashSet::new();
    for (src, dst, kind, weight) in edges {
        if src != target && dst != target {
            continue;
        }
        let Ok(edge_type) = kind.parse::<EdgeType>() else {
            continue;
        };
        if src.is_empty() || dst.is_empty() || !weight.is_finite() || !(0.0..=1.0).contains(weight)
        {
            continue;
        }
        if !seen.insert((src.clone(), dst.clone(), kind.clone())) {
            continue;
        }
        endpoints.insert(src.clone());
        endpoints.insert(dst.clone());
        candidates.push((src.clone(), dst.clone(), kind.clone(), edge_type, *weight));
    }
    // Do not scan local relations at all for an unrelated cloud graph.
    if candidates.is_empty() {
        return Ok(EdgeImportResult::default());
    }
    let mut params = BTreeMap::new();
    params.insert("target".to_string(), DataValue::Str(target.into()));
    // No @ modifier: a retracted key is a local decision too. Non-local soft
    // links are distinct keys and must not suppress a real local edge.
    let rows = store.db_ref().run_script(r#"
        ?[src, dst, edge_type] := *edge{src, dst, edge_type, dst_store}, src = $target, dst_store = 'local'
        ?[src, dst, edge_type] := *edge{src, dst, edge_type, dst_store}, dst = $target, dst_store = 'local'
    "#, params, ScriptMutability::Immutable)?;
    let mut known: HashSet<(String, String, String)> = rows
        .rows
        .iter()
        .filter_map(|r| {
            Some((
                r.first()?.get_str()?.into(),
                r.get(1)?.get_str()?.into(),
                r.get(2)?.get_str()?.into(),
            ))
        })
        .collect();
    let mut result = EdgeImportResult::default();
    candidates.retain(|(src, dst, kind, _, _)| {
        if known.contains(&(src.clone(), dst.clone(), kind.clone())) {
            result.kept += 1;
            false
        } else {
            true
        }
    });
    if candidates.is_empty() {
        return Ok(result);
    }
    let ids: Vec<String> = endpoints.into_iter().collect();
    let present = existing_node_ids(store, &ids)?;
    for (src, dst, kind, edge_type, weight) in candidates {
        if cancelled() {
            return Err(crate::Error::Invalid(
                "edge import cancelled; partial import retained".into(),
            ));
        }
        if !present.contains(&src) || !present.contains(&dst) {
            continue;
        }
        store.scoped(Some(initiative), |s| {
            link_with_weight(s, &src, &dst, edge_type, weight)
        })?;
        known.insert((src, dst, kind));
        result.linked += 1;
    }
    Ok(result)
}

/// Batch, parameterised, live-node existence lookup. It deliberately ignores
/// active scope, like the old node_brief_by_id check, and never reads bodies.
pub fn existing_node_ids(store: &Store, ids: &[String]) -> Result<HashSet<String>> {
    let mut present = HashSet::new();
    for chunk in ids.chunks(256) {
        let mut params = BTreeMap::new();
        params.insert(
            "ids".to_string(),
            DataValue::List(
                chunk
                    .iter()
                    .map(|id| DataValue::Str(id.as_str().into()))
                    .collect(),
            ),
        );
        let rows = store.db_ref().run_script(
            "candidates[id] := id in $ids
?[id] := candidates[id], *node{id @ 'NOW'}",
            params,
            ScriptMutability::Immutable,
        )?;
        for row in rows.rows {
            if let Some(id) = row.first().and_then(|v| v.get_str()) {
                present.insert(id.to_string());
            }
        }
    }
    Ok(present)
}

#[cfg(test)]
mod tests {
    use super::{existing_node_ids, restore_cloud_edges};
    use crate::{
        EdgeType, EpisodeKind, Significance, Store, edges_of, forget, link_with_weight, unlink,
        write_episode,
    };
    fn node(s: &Store, name: &str) -> String {
        s.scoped(Some("alpha"), |s| {
            write_episode(s, EpisodeKind::Observation, Significance::Low, name, "body")
        })
        .unwrap()
    }
    #[test]
    fn cancelled_import_does_not_mutate() {
        let s = Store::open_in_memory().unwrap();
        let a = node(&s, "a");
        let b = node(&s, "b");
        let edges = vec![(a.clone(), b, "refers_to".into(), 0.8)];
        assert!(super::restore_cloud_edges_cancellable(&s, "alpha", &a, &edges, || true).is_err());
        assert!(edges_of(&s, &a).unwrap().is_empty());
    }
    #[test]
    fn cancellation_between_mutations_retains_valid_partial_import() {
        use std::cell::Cell;
        let s = Store::open_in_memory().unwrap();
        let a = node(&s, "a");
        let b = node(&s, "b");
        let c = node(&s, "c");
        let edges = vec![
            (a.clone(), b, "refers_to".into(), 0.8),
            (a.clone(), c, "refers_to".into(), 0.6),
        ];
        let checks = Cell::new(0);
        let result = super::restore_cloud_edges_cancellable(&s, "alpha", &a, &edges, || {
            let count = checks.get() + 1;
            checks.set(count);
            count >= 3
        });
        assert!(result.is_err());
        assert_eq!(edges_of(&s, &a).unwrap().len(), 1);
        // An explicit resume fills only the missing edge.
        let resumed = restore_cloud_edges(&s, "alpha", &a, &edges).unwrap();
        assert_eq!((resumed.linked, resumed.kept), (1, 1));
    }
    #[test]
    fn import_uses_requested_scope_and_later_scoped_reads_keep_theirs() {
        let s = Store::open_in_memory().unwrap();
        let a = node(&s, "a");
        let b = node(&s, "b");
        s.use_initiative("beta");
        let edges = vec![(a.clone(), b, "refers_to".into(), 0.8)];
        restore_cloud_edges(&s, "alpha", &a, &edges).unwrap();
        assert_eq!(s.current_initiative().as_deref(), Some("alpha"));
        s.scoped(Some("beta"), |s| {
            assert_eq!(s.current_initiative().as_deref(), Some("beta"));
        });
    }
    #[test]
    fn preserves_weights_retractions_and_idempotence() {
        let s = Store::open_in_memory().unwrap();
        let a = node(&s, "a");
        let b = node(&s, "b");
        let c = node(&s, "c");
        let edges = vec![
            (a.clone(), b.clone(), "refers_to".into(), 0.8),
            (c.clone(), a.clone(), "refers_to".into(), 0.6),
        ];
        let first = restore_cloud_edges(&s, "alpha", &a, &edges).unwrap();
        assert_eq!((first.linked, first.kept), (2, 0));
        s.scoped(Some("alpha"), |s| {
            link_with_weight(s, &a, &b, EdgeType::RefersTo, 0.2)
        })
        .unwrap();
        s.scoped(Some("alpha"), |s| unlink(s, &c, &a, EdgeType::RefersTo))
            .unwrap();
        let again = restore_cloud_edges(&s, "alpha", &a, &edges).unwrap();
        assert_eq!((again.linked, again.kept), (0, 2));
        let live = edges_of(&s, &a).unwrap();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].3, 0.2);
    }
    #[test]
    fn skips_missing_retracted_unknown_and_unrelated_edges() {
        let s = Store::open_in_memory().unwrap();
        let a = node(&s, "a");
        let b = node(&s, "b");
        let c = node(&s, "c");
        forget(&s, &c).unwrap();
        let edges = vec![
            (a.clone(), "missing".into(), "refers_to".into(), 0.9),
            (a.clone(), c.clone(), "refers_to".into(), 0.9),
            (a.clone(), b.clone(), "future_kind".into(), 0.9),
            (b.clone(), b.clone(), "refers_to".into(), 0.9),
        ];
        assert_eq!(
            restore_cloud_edges(&s, "alpha", &a, &edges).unwrap().linked,
            0
        );
        assert_eq!(
            existing_node_ids(&s, &[a.clone(), b.clone(), c])
                .unwrap()
                .len(),
            2
        );
        // Scope is restored per mutation; reads from a second initiative cannot
        // accidentally inherit alpha from an unguarded set-then-write sequence.
        assert!(
            s.scoped(Some("beta"), |s| crate::recall_id_by_name(s, "a"))
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn unrelated_scale_costs_no_local_queries_and_duplicates_keep_first_weight() {
        let s = Store::open_in_memory().unwrap();
        let a = node(&s, "a");
        let b = node(&s, "b");
        let mut edges: Vec<_> = (0..2000)
            .map(|i| (format!("x{i}"), format!("y{i}"), "refers_to".into(), 0.9))
            .collect();
        edges.push((a.clone(), b.clone(), "refers_to".into(), 0.4));
        edges.push((a.clone(), b.clone(), "refers_to".into(), 0.8));
        assert_eq!(
            restore_cloud_edges(&s, "alpha", &a, &edges).unwrap().linked,
            1
        );
        assert_eq!(edges_of(&s, &a).unwrap()[0].3, 0.4);
    }
}
