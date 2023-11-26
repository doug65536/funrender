#pragma once
#include <cstdlib>
#include <cstdint>
#include <list>
#include <map>
#include <array>
#include <type_traits>

template<typename T, size_t log2_bucket_sz = 7>
class pool
{
public:
    using bucket_shift = std::integral_constant<
        size_t, log2_bucket_sz>;
    using bucket_sz = std::integral_constant<
        size_t, size_t(1) << log2_bucket_sz>;
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    struct pool_item_deleter {
        pool_item_deleter(pool *owner_) : owner(owner_) {}
        void operator()(void *p)
        {
            owner->recycle(p);
        }
        pool *owner{};
    };

    using item_ptr = std::unique_ptr<T, pool_item_deleter>;

    template<typename ...Args>
    item_ptr alloc(Args&& ...args)
    {
        void *item_memory;

        // Fastpath
        if (free_list) {
            item_memory = reinterpret_cast<void*>(free_list);
            free_slot *next_free = free_list->next;
            free_list->~free_slot();
        } else {
            // Create a bucket if needed
            if (buckets.empty() ||
                    buckets.back().bump_alloc == bucket_sz::value) {
                // Create a bucket
                buckets.emplace_back();
                capacity_ += bucket_sz::value;
            }

            // Take item
            size_t index = buckets.back().bump_alloc++;

            item_memory = buckets.back().storage[index].data;

        }
        T *item = new (item_memory) T(std::forward<Args>(args)...);

        ++used_;

        return item_ptr{item, this};
    }

    void recycle(void *p)
    {
        T *item = reinterpret_cast<T*>(p);
        item->~T();
        free_slot *slot = new (item) free_slot{};
        slot->next = free_list;
        free_list = slot;
        --used_;
    }

    size_t capacity() const
    {
        return capacity_;
    }

    size_t avail() const
    {
        return capacity_ - used_;
    }

    size_t used() const
    {
        return used_;
    }

    void reserve(size_t capacity)
    {
        if (capacity_ >= capacity)
            return;

        size_t rounded_capacity = (capacity + (bucket_sz::value - 1)) /
            bucket_sz::value;
        while (capacity_ < rounded_capacity) {
            buckets.emplace_back();
            capacity_ += bucket_sz::value;
        }
    }

    // Try to return some buckets to the OS memory pool
    // if the usage threshold is below the provided percentage
    void trim(int threshold_percent)
    {
        if (used_ == 0) {
            buckets.clear();
            return;
        }

        if (threshold_percent != 0) {
            if (used_ * 100 / capacity_ >= threshold_percent)
                return;
        }

        using bucket_to_bucket_list_it =
            std::map<void *, typename bucket_list::iterator>;
        bucket_to_bucket_list_it bucket_list_its;
        for (typename bucket_list::iterator it = buckets.begin();
                it != buckets.end(); ++it)
            bucket_list_its.emplace(it->storage.data(), it);

        // Walk the free list
        using free_counts_map =
            std::map<typename bucket_list::iterator, size_t>;
        free_counts_map free_counts;
        for (free_slot *item = free_list; item; item = item->next) {
            typename bucket_to_bucket_list_it::iterator it =
                bucket_list_its.lower_bound((void*)item);
            assert(it != bucket_list_its.end());
            ++free_counts[it];
        }

        // Walk the free counts
        for (typename free_counts_map::iterator it = free_counts.begin();
                it != free_counts.end(); ++it) {
            if (it->second == bucket_sz::value)
                buckets.erase(it->first);
        }
    }
private:
    struct free_slot {
        free_slot *next{};
    };

    using slot = std::aligned_storage_t<
        std::max(sizeof(T), sizeof(free_slot)),
        std::max(alignof(T), alignof(free_slot))>;

    using bucket_storage = std::array<slot, bucket_sz::value>;

    struct bucket {
        size_t bump_alloc{};
        bucket_storage storage;
    };

    using bucket_list = std::list<bucket>;
    bucket_list buckets;

    free_slot *free_list{};

    size_t capacity_{};
    size_t used_{};
};
