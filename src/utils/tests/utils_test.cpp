#include <utils/get_if.hpp>
#include <utils/utils.hpp>

#include <gtest/gtest.h>

#include <type_traits>

using ScopeGuard = utils::ScopeGuard<void (*)()>;

static_assert(
    !std::is_default_constructible_v<ScopeGuard>,
    "one-time");
static_assert(
    std::is_nothrow_constructible_v<
        ScopeGuard,
        void (*)()>,
    "one-time");

// NOLINTBEGIN(readability-convert-member-functions-to-static)

TEST(
    getIf,
    PointerLike)
{
    static_assert(!utils::PointerLike<int>);
    static_assert(!utils::PointerLike<int &>);
    static_assert(!utils::PointerLike<std::nullptr_t>);

    static_assert(!utils::PointerLike<std::optional<int>>);
    static_assert(utils::PointerLike<std::optional<int> &>);

    static_assert(utils::PointerLike<std::shared_ptr<int>>);
    static_assert(utils::PointerLike<std::shared_ptr<int> &>);

    static_assert(utils::PointerLike<std::unique_ptr<int>>);
    static_assert(utils::PointerLike<std::unique_ptr<int> &>);

    static_assert(utils::PointerLike<int *>);
    static_assert(utils::PointerLike<int **>);
    static_assert(utils::PointerLike<int *&>);
}

TEST(
    getIf,
    Basic)
{
    struct B
    {
    };
    struct A
    {
        B b;
    };

    {
        A a{};
        EXPECT_EQ(utils::getIf(a, &A::b), &a.b);
        EXPECT_EQ(GET_IF(a, b), &a.b);
    }

    {
        A a{};
        EXPECT_EQ(utils::getIf(std::ref(a), &A::b), &a.b);
    }

    {
        A a{};
        EXPECT_EQ(utils::getIf(&a, &A::b), &a.b);
        EXPECT_EQ(GET_IF(&a, b), &a.b);
    }

    {
        A * a = nullptr;
        EXPECT_EQ(utils::getIf(a, &A::b), static_cast<B *>(nullptr));
        EXPECT_EQ(GET_IF(a, b), static_cast<B *>(nullptr));
    }

    {
        auto a = std::make_optional<A>();
        EXPECT_EQ(utils::getIf(a, &A::b), &a->b);
        EXPECT_EQ(GET_IF(a, b), &a->b);
    }

    {
        std::optional<A> a;
        EXPECT_EQ(utils::getIf(a, &A::b), static_cast<B *>(nullptr));
        EXPECT_EQ(GET_IF(a, b), static_cast<B *>(nullptr));
    }

    {
        auto a = std::make_unique<A>();
        EXPECT_EQ(utils::getIf(a, &A::b), &a->b);
        EXPECT_EQ(GET_IF(a, b), &a->b);
    }

    {
        std::unique_ptr<A> a;
        EXPECT_EQ(utils::getIf(a, &A::b), static_cast<B *>(nullptr));
        EXPECT_EQ(GET_IF(a, b), static_cast<B *>(nullptr));
    }

    {
        auto a = std::make_shared<A>();
        EXPECT_EQ(utils::getIf(a, &A::b), &a->b);
        EXPECT_EQ(GET_IF(a, b), &a->b);
    }

    {
        const std::shared_ptr<A> a;
        EXPECT_EQ(utils::getIf(a, &A::b), static_cast<B *>(nullptr));
        EXPECT_EQ(GET_IF(a, b), static_cast<B *>(nullptr));
    }
}

TEST(
    getIf,
    Chain)
{
    struct C
    {
    };
    struct B
    {
        std::unique_ptr<C> c = std::make_unique<C>();
    };
    struct A
    {
        std::shared_ptr<B> b = std::make_shared<B>();
    };

    auto a = std::make_optional<A>();
    EXPECT_EQ(utils::getIf(a, &A::b, &B::c), a->b->c.get());
    EXPECT_EQ(GET_IF(a, b, c), a->b->c.get());
    EXPECT_EQ(utils::getIf(a, &A::b), a->b.get());
    EXPECT_EQ(GET_IF(a, b), a->b.get());
    EXPECT_EQ(utils::getIf(a), &*a);
    EXPECT_EQ(GET_IF(a), &*a);
    a->b->c.reset();
    EXPECT_EQ(utils::getIf(a, &A::b, &B::c), static_cast<C *>(nullptr));
    EXPECT_EQ(GET_IF(a, b, c), static_cast<C *>(nullptr));
    EXPECT_EQ(utils::getIf(a, &A::b), a->b.get());
    EXPECT_EQ(GET_IF(a, b), a->b.get());
    EXPECT_EQ(utils::getIf(a), &*a);
    EXPECT_EQ(GET_IF(a), &*a);
    a->b.reset();
    EXPECT_EQ(utils::getIf(a, &A::b, &B::c), static_cast<C *>(nullptr));
    EXPECT_EQ(GET_IF(a, b, c), static_cast<C *>(nullptr));
    EXPECT_EQ(utils::getIf(a, &A::b), static_cast<B *>(nullptr));
    EXPECT_EQ(GET_IF(a, b), static_cast<B *>(nullptr));
    EXPECT_EQ(utils::getIf(a), &*a);
    EXPECT_EQ(GET_IF(a), &*a);
    a.reset();
    EXPECT_EQ(utils::getIf(a, &A::b, &B::c), static_cast<C *>(nullptr));
    EXPECT_EQ(GET_IF(a, b, c), static_cast<C *>(nullptr));
    EXPECT_EQ(utils::getIf(a, &A::b), static_cast<B *>(nullptr));
    EXPECT_EQ(GET_IF(a, b), static_cast<B *>(nullptr));
    EXPECT_EQ(utils::getIf(a), static_cast<A *>(nullptr));
    EXPECT_EQ(GET_IF(a), static_cast<A *>(nullptr));
}

TEST(
    getIf,
    DoubleIndirection)
{
    struct B
    {
    };
    struct A
    {
        std::optional<std::optional<B>> b = B{};
    };

    {
        A a;
        EXPECT_EQ(utils::getIf(a, &A::b), &**a.b);
        EXPECT_EQ(GET_IF(a, b), &**a.b);
        a.b->reset();
        EXPECT_EQ(utils::getIf(a, &A::b), static_cast<B *>(nullptr));
        EXPECT_EQ(GET_IF(a, b), static_cast<B *>(nullptr));
    }

    {
        A a;
        EXPECT_EQ(utils::getIf(a, &A::b), &**a.b);
        EXPECT_EQ(GET_IF(a, b), &**a.b);
        a.b.reset();
        EXPECT_EQ(utils::getIf(a, &A::b), static_cast<B *>(nullptr));
        EXPECT_EQ(GET_IF(a, b), static_cast<B *>(nullptr));
    }
}

TEST(
    getIf,
    Unary)
{
    struct A
    {
    };

    A a;

    for (int i = 0; i < 4; ++i) {
        auto p = std::make_optional(std::make_unique<std::shared_ptr<A *>>(std::make_shared<A *>(&a)));
        ASSERT_EQ(utils::getIf(p), &a);
        ASSERT_EQ(GET_IF(p), &a);
        switch (i) {
        case 0: {
            ***p = nullptr;
            ASSERT_TRUE(p && *p && **p && !***p);
            break;
        }
        case 1: {
            (**p).reset();
            ASSERT_TRUE(p && *p && !**p);
            break;
        }
        case 2: {
            p->reset();
            ASSERT_TRUE(p && !*p);
            break;
        }
        case 3: {
            p.reset();
            ASSERT_TRUE(!p);
            break;
        }
        }
        EXPECT_EQ(utils::getIf(p), static_cast<A *>(nullptr));
        EXPECT_EQ(GET_IF(p), static_cast<A *>(nullptr));
    }
    for (int i = 0; i < 3; ++i) {
        auto * r = &a;
        auto * q = &r;
        auto * p = &q;
        ASSERT_EQ(utils::getIf(p), &a);
        ASSERT_EQ(GET_IF(p), &a);
        switch (i) {
        case 0: {
            r = nullptr;
            break;
        }
        case 1: {
            q = nullptr;
            break;
        }
        case 2: {
            p = nullptr;
            break;
        }
        }
        EXPECT_EQ(utils::getIf(p), static_cast<A *>(nullptr));
        EXPECT_EQ(GET_IF(p), static_cast<A *>(nullptr));
    }
}

TEST(
    getIf,
    RawPtr)
{
    struct B
    {
    };
    struct A
    {
        B * b;
    };

    B b;
    A a{&b};
    EXPECT_EQ(utils::getIf(a, &A::b), &b);
    EXPECT_EQ(GET_IF(a, b), &b);
}

TEST(
    getIf,
    FreeFunction)
{
    struct B
    {
    };
    struct A
    {
        B b;
    };

    auto g = [](A & a) -> B &
    {
        return a.b;
    };
    A a{};
    EXPECT_EQ(utils::getIf(a, g), &a.b);
    EXPECT_EQ(utils::getIf(a, std::ref(g)), &a.b);

    using FuncPtr = B & (*)(A &);
    FuncPtr f = g;
    EXPECT_EQ(utils::getIf(a, f), &a.b);
    EXPECT_EQ(utils::getIf(a, std::ref(f)), &a.b);
}

TEST(
    getIf,
    MemberFunction)
{
    struct A
    {
        int b = 123;
        int & f()
        {
            return b;
        }
        int & g(int & c)
        {
            return c;
        }
    };

    A a{};

    {
        EXPECT_EQ(utils::getIf(a, &A::f), &a.b);
        EXPECT_EQ(GET_IF(a, f()), &a.b);
    }
    {
        int c = 321;
        EXPECT_EQ(GET_IF(a, g(c)), &c);
    }
}

TEST(
    getIf,
    Capture)
{
    struct A
    {
        int & g(int & c)
        {
            return c;
        };
    };

    A a{};

    {
        int c = 321;
        EXPECT_EQ(GET_IF(a, g(c)), &c);
    }
}

TEST(
    getIf,
    PerfectForwarding)
{
    struct C
    {
    };
    struct B
    {
    };
    struct A
    {
        C c[4];

        C & operator()(B &)
        {
            return c[0];
        }
        C & operator()(const B &)
        {
            return c[1];
        }
        C & operator()(B &&)
        {
            return c[2];
        }
        C & operator()(const B &&)
        {
            return c[3];
        }
    };

    A a{};
    B b;
    EXPECT_EQ(utils::getIf(b, a), a.c + 0);
    EXPECT_EQ(utils::getIf(std::as_const(b), a), a.c + 1);
    EXPECT_EQ(utils::getIf(std::move(b), a), a.c + 2);
    EXPECT_EQ(utils::getIf(std::move(std::as_const(b)), a), a.c + 3);
}

// NOLINTEND(readability-convert-member-functions-to-static)
